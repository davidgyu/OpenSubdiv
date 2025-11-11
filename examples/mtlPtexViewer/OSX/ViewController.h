//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#import <AppKit/AppKit.h>
#import <MetalKit/MetalKit.h>
#import "../mtlPtexViewer.h"
#import "../../common/mtlHud.h"

@class ViewController;

@interface OSDView : MTKView {
    @public
    MTLhud hud;
};

@property (nonatomic) ViewController* controller;
@end

@interface ViewController : NSViewController<MTKViewDelegate, OSDRendererDelegate>
@property (weak) IBOutlet OSDView *view;
@property (nonatomic) OSDRenderer* osdRenderer;

@property (weak) IBOutlet NSTextField *frameTimeLabel;
@property (weak) IBOutlet NSButton *wireframeCheckbox;
@property (weak) IBOutlet NSButton *singleCreaseCheckbox;
@property (weak) IBOutlet NSButton *patchIndexCheckbox;
@property (weak) IBOutlet NSButton *patchClipCullingCheckbox;
@property (weak) IBOutlet NSButton *backfaceCullingCheckbox;
@property (weak) IBOutlet NSButton *backpatchCullingCheckbox;
@property (weak) IBOutlet NSButton *screenspaceTessellationCheckbox;
@property (weak) IBOutlet NSPopUpButton *modelPopup;
@property (weak) IBOutlet NSPopUpButton *refinementLevelPopup;
@property (weak) IBOutlet NSPopUpButton *tessellationLevelPopup;
@property (weak) IBOutlet NSPopUpButton *displacementModePopup;
@property (weak) IBOutlet NSPopUpButton *normalModePopup;
@property (weak) IBOutlet NSPopUpButton *colorModePopup;
@property (weak) IBOutlet NSSlider *displacementScaleSlider;
@property (weak) IBOutlet NSSlider *mipmapBiasSlider;


@end
