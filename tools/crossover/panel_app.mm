// FSR 4 panel: a small Dock application (dist/crossover/FSR4 Panel.app) that
// shows what the CrossOver hook is doing and sets the RCAS sharpness. It is a
// separate process; the hook publishes dist/crossover/status.conf at 2 Hz and
// reloads dist/crossover/panel.conf (written here) when its mtime changes.
// The window is a non-activating floating panel, so clicking the slider does
// not take focus away from the game.
//
//   FSR4 Panel [--root <repo>] [--snapshot <png>] [--make-icon <dir>]
#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>
#include <sys/stat.h>
#include <string>
#include <map>
#include <mach-o/dyld.h>

struct Status {
    bool ok = false, active = false;
    std::string reason, model, preset;
    unsigned rw = 0, rh = 0, uw = 0, uh = 0;
    double frameMs = 0, t = 0;
    unsigned long long replaced = 0, stubs = 0;
};

static std::string g_root;
static std::string dist_path(const char *name) { return g_root + "/dist/crossover/" + name; }

static std::map<std::string, std::string> read_kv(const std::string &path) {
    std::map<std::string, std::string> kv;
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return kv;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *eq = strchr(line, '=');
        if (!eq) continue;
        *eq = 0;
        std::string v(eq + 1);
        while (!v.empty() && (v.back() == '\n' || v.back() == '\r')) v.pop_back();
        kv[line] = v;
    }
    fclose(f);
    return kv;
}

static Status read_status() {
    Status s;
    auto kv = read_kv(dist_path("status.conf"));
    if (kv.empty()) return s;
    s.ok = true;
    s.active = atoi(kv["active"].c_str()) != 0;
    s.reason = kv["reason"]; s.model = kv["model"]; s.preset = kv["preset"];
    s.rw = atoi(kv["rw"].c_str()); s.rh = atoi(kv["rh"].c_str());
    s.uw = atoi(kv["uw"].c_str()); s.uh = atoi(kv["uh"].c_str());
    s.frameMs = atof(kv["frame_ms"].c_str());
    s.t = atof(kv["t"].c_str());
    s.replaced = strtoull(kv["replaced"].c_str(), nullptr, 10);
    s.stubs = strtoull(kv["stubs"].c_str(), nullptr, 10);
    return s;
}

static void read_settings(float &sharp, bool &rcas) {
    sharp = 0.4f; rcas = true;
    auto kv = read_kv(dist_path("panel.conf"));
    if (kv.count("sharpness")) sharp = (float)atof(kv["sharpness"].c_str());
    if (kv.count("rcas")) rcas = atoi(kv["rcas"].c_str()) != 0;
    if (sharp < 0) sharp = 0; if (sharp > 1) sharp = 1;
}

static void write_settings(float sharp, bool rcas) {
    const std::string p = dist_path("panel.conf"), tmp = p + ".tmp";
    FILE *f = fopen(tmp.c_str(), "w");
    if (!f) return;
    fprintf(f, "sharpness=%.3f\nrcas=%d\n", sharp, rcas ? 1 : 0);
    fclose(f);
    rename(tmp.c_str(), p.c_str());
}

@interface PanelApp : NSObject <NSApplicationDelegate>
@property(nonatomic, strong) NSPanel *panel;
@property(nonatomic, strong) NSTextField *status, *res, *frame, *sharpVal;
@property(nonatomic, strong) NSSlider *slider;
@property(nonatomic, strong) NSButton *rcas;
@property(nonatomic, strong) NSTimer *timer;
@end

@implementation PanelApp

static NSTextField *label(NSString *s, CGFloat x, CGFloat y, CGFloat w, BOOL bold) {
    NSTextField *t = [NSTextField labelWithString:s];
    t.frame = NSMakeRect(x, y, w, 18);
    t.font = bold ? [NSFont boldSystemFontOfSize:12] : [NSFont systemFontOfSize:12];
    t.textColor = NSColor.whiteColor;
    t.lineBreakMode = NSLineBreakByTruncatingTail;
    return t;
}

- (void)build {
    const CGFloat W = 380, H = 158;
    NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskUtilityWindow
                     | NSWindowStyleMaskNonactivatingPanel | NSWindowStyleMaskHUDWindow;
    NSPanel *p = [[NSPanel alloc] initWithContentRect:NSMakeRect(40, 80, W, H)
                                            styleMask:style backing:NSBackingStoreBuffered defer:NO];
    p.title = @"FSR 4";
    p.level = NSStatusWindowLevel;
    p.collectionBehavior = NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorFullScreenAuxiliary
                         | NSWindowCollectionBehaviorStationary;
    p.hidesOnDeactivate = NO;
    p.floatingPanel = YES;
    p.becomesKeyOnlyIfNeeded = YES;
    p.releasedWhenClosed = NO;
    p.alphaValue = 0.94;
    p.frameAutosaveName = @"FSR4Panel";
    NSView *v = p.contentView;

    CGFloat y = H - 26;
    [v addSubview:label(@"FSR 4 вместо DLSS (CrossOver)", 12, y, W - 24, YES)]; y -= 22;
    self.status = label(@"", 12, y, W - 24, NO); [v addSubview:self.status]; y -= 20;
    self.res = label(@"", 12, y, W - 24, NO); [v addSubview:self.res]; y -= 20;
    self.frame = label(@"", 12, y, W - 24, NO); [v addSubview:self.frame]; y -= 30;

    float sharp; bool rc;
    read_settings(sharp, rc);
    [v addSubview:label(@"Резкость", 12, y, 70, NO)];
    self.slider = [NSSlider sliderWithValue:sharp minValue:0.0 maxValue:1.0 target:self action:@selector(changed:)];
    self.slider.frame = NSMakeRect(84, y - 2, W - 84 - 12 - 50, 22);
    self.slider.continuous = YES;
    [v addSubview:self.slider];
    self.sharpVal = label(@"", W - 12 - 46, y, 46, NO);
    self.sharpVal.alignment = NSTextAlignmentRight;
    [v addSubview:self.sharpVal]; y -= 26;
    self.rcas = [NSButton checkboxWithTitle:@"RCAS (шарпинг FSR)" target:self action:@selector(changed:)];
    self.rcas.frame = NSMakeRect(12, y, W - 24, 18);
    self.rcas.state = rc ? NSControlStateValueOn : NSControlStateValueOff;
    self.rcas.attributedTitle = [[NSAttributedString alloc] initWithString:self.rcas.title
        attributes:@{NSForegroundColorAttributeName: NSColor.whiteColor, NSFontAttributeName: [NSFont systemFontOfSize:12]}];
    [v addSubview:self.rcas];

    self.panel = p;
    [self refresh];
    [p orderFrontRegardless];
    self.timer = [NSTimer scheduledTimerWithTimeInterval:0.5 target:self selector:@selector(tick) userInfo:nil repeats:YES];
    [[NSRunLoop mainRunLoop] addTimer:self.timer forMode:NSRunLoopCommonModes];
}

- (void)changed:(id)sender {
    write_settings((float)self.slider.doubleValue, self.rcas.state == NSControlStateValueOn);
    [self refresh];
}

- (void)tick {
    if (self.panel.visible) [self refresh];
}

- (void)refresh {
    Status st = read_status();
    const double age = st.ok ? (double)time(NULL) - st.t : 1e9;
    if (!st.ok)
        self.status.stringValue = @"Состояние: нет данных — игра с хуком ещё не запускалась";
    else if (age > 3.0)
        self.status.stringValue = [NSString stringWithFormat:@"Состояние: нет кадров %.0f с (игра закрыта или без апскейла)", age];
    else if (st.active)
        self.status.stringValue = [NSString stringWithFormat:@"Состояние: FSR 4 активен (%s, %s)", st.model.c_str(), st.preset.c_str()];
    else
        self.status.stringValue = [NSString stringWithFormat:@"Состояние: неактивен — %@",
                                   st.reason.empty() ? @"ещё не было кадров" : [NSString stringWithUTF8String:st.reason.c_str()]];
    if (st.ok && st.rw)
        self.res.stringValue = [NSString stringWithFormat:@"Апскейл: %u×%u → %u×%u  (кадров %llu, заглушек %llu)",
                                st.rw, st.rh, st.uw, st.uh, st.replaced, st.stubs];
    else
        self.res.stringValue = @"Апскейл: —";
    if (st.ok && st.frameMs > 0 && age <= 3.0)
        self.frame.stringValue = [NSString stringWithFormat:@"Кадр игры: %.1f ms (%.0f fps, среднее)", st.frameMs, 1000.0 / st.frameMs];
    else
        self.frame.stringValue = @"Кадр игры: —";
    float s; bool rc;
    read_settings(s, rc);
    self.sharpVal.stringValue = [NSString stringWithFormat:@"%.2f", s];
    if (!self.slider.highlighted && fabs(self.slider.doubleValue - s) > 1e-3) self.slider.doubleValue = s;
    NSControlStateValue want = rc ? NSControlStateValueOn : NSControlStateValueOff;
    if (self.rcas.state != want) self.rcas.state = want;
}

- (void)applicationDidFinishLaunching:(NSNotification *)n {
    [self build];
}

- (BOOL)applicationShouldHandleReopen:(NSApplication *)sender hasVisibleWindows:(BOOL)flag {
    [self.panel orderFrontRegardless];   // Dock click re-shows a closed panel
    return NO;
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender { return NO; }

// test aid: content view to PNG (the HUD material is a window property, so
// the view gets a dark layer background for the capture)
- (void)snapshot:(NSString *)path {
    NSView *v = self.panel.contentView;
    v.wantsLayer = YES;
    v.layer.backgroundColor = CGColorCreateGenericGray(0.16, 1.0);
    [v displayIfNeeded];
    NSBitmapImageRep *rep = [v bitmapImageRepForCachingDisplayInRect:v.bounds];
    [v cacheDisplayInRect:v.bounds toBitmapImageRep:rep];
    [[rep representationUsingType:NSBitmapImageFileTypePNG properties:@{}] writeToFile:path atomically:YES];
    printf("panel %.0fx%.0f written to %s\n", v.bounds.size.width, v.bounds.size.height, path.UTF8String);
}

@end

// icon: rounded dark tile with "FSR 4", rendered with CoreGraphics (no window needed)
static void make_icon_png(const char *path, int size) {
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef c = CGBitmapContextCreate(NULL, size, size, 8, 0, cs, kCGImageAlphaPremultipliedLast);
    const CGFloat r = size * 0.22, inset = size * 0.04;
    CGRect rect = CGRectMake(inset, inset, size - 2 * inset, size - 2 * inset);
    CGPathRef rr = CGPathCreateWithRoundedRect(rect, r, r, NULL);
    CGContextAddPath(c, rr);
    CGContextClip(c);
    CGFloat comps[8] = {0.55, 0.12, 0.12, 1.0, 0.20, 0.05, 0.05, 1.0};
    CGGradientRef g = CGGradientCreateWithColorComponents(cs, comps, NULL, 2);
    CGContextDrawLinearGradient(c, g, CGPointMake(0, size), CGPointMake(0, 0), 0);
    CGGradientRelease(g);
    CGPathRelease(rr);
    NSFont *f = [NSFont boldSystemFontOfSize:size * 0.30];
    NSDictionary *attrs = @{NSFontAttributeName: f, NSForegroundColorAttributeName: NSColor.whiteColor};
    NSAttributedString *s1 = [[NSAttributedString alloc] initWithString:@"FSR" attributes:attrs];
    NSAttributedString *s2 = [[NSAttributedString alloc] initWithString:@"4" attributes:attrs];
    NSGraphicsContext *gc = [NSGraphicsContext graphicsContextWithCGContext:c flipped:NO];
    [NSGraphicsContext saveGraphicsState];
    [NSGraphicsContext setCurrentContext:gc];
    NSSize a = s1.size, b = s2.size;
    [s1 drawAtPoint:NSMakePoint((size - a.width) / 2, size * 0.50)];
    [s2 drawAtPoint:NSMakePoint((size - b.width) / 2, size * 0.16)];
    [NSGraphicsContext restoreGraphicsState];
    CGImageRef img = CGBitmapContextCreateImage(c);
    NSBitmapImageRep *rep = [[NSBitmapImageRep alloc] initWithCGImage:img];
    [[rep representationUsingType:NSBitmapImageFileTypePNG properties:@{}] writeToFile:[NSString stringWithUTF8String:path] atomically:YES];
    CGImageRelease(img);
    CGContextRelease(c);
    CGColorSpaceRelease(cs);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *snap = nullptr, *icondir = nullptr;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--root") && i + 1 < argc) g_root = argv[++i];
            else if (!strcmp(argv[i], "--snapshot") && i + 1 < argc) snap = argv[++i];
            else if (!strcmp(argv[i], "--make-icon") && i + 1 < argc) icondir = argv[++i];
        }
        if (icondir) {
            const int sizes[] = {16, 32, 64, 128, 256, 512, 1024};
            for (int sz : sizes) {
                char p[1024];
                if (sz <= 512) { snprintf(p, sizeof p, "%s/icon_%dx%d.png", icondir, sz, sz); make_icon_png(p, sz); }
                if (sz >= 32) { snprintf(p, sizeof p, "%s/icon_%dx%d@2x.png", icondir, sz / 2, sz / 2); make_icon_png(p, sz); }
            }
            return 0;
        }
        if (g_root.empty()) {
            if (const char *e = getenv("FSR4_ROOT")) g_root = e;
        }
        if (g_root.empty()) {
            // dist/crossover/FSR4 Panel.app/Contents/MacOS/<exe> -> repo root
            char buf[4096]; uint32_t sz = sizeof buf;
            if (_NSGetExecutablePath(buf, &sz) == 0) {
                std::string p = buf;
                for (int i = 0; i < 5; i++) p = p.substr(0, p.rfind('/'));
                g_root = p;
            }
        }
        NSApplication *app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        // minimal menu so Cmd+Q works
        NSMenu *bar = [NSMenu new];
        NSMenuItem *appItem = [NSMenuItem new];
        [bar addItem:appItem];
        NSMenu *appMenu = [NSMenu new];
        [appMenu addItemWithTitle:@"Завершить FSR 4" action:@selector(terminate:) keyEquivalent:@"q"];
        appItem.submenu = appMenu;
        app.mainMenu = bar;
        PanelApp *pa = [PanelApp new];
        app.delegate = pa;
        if (snap) {
            NSString *path = [NSString stringWithUTF8String:snap];
            dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(1.0 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
                [pa snapshot:path];
                exit(0);
            });
        }
        [app run];
    }
    return 0;
}
