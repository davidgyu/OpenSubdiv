//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//


#import <AppKit/AppKit.h>
#import <MetalKit/MetalKit.h>
#import "../mtlViewer.h"
#import "../../common/mtlHud.h"
@class ViewController;

@interface OSDView : MTKView {
    @public
    MTLhud hud;
}
@property (nonatomic) ViewController* controller;
@end

@interface ViewController : NSViewController<MTKViewDelegate, OSDRendererDelegate>
@property (weak) IBOutlet OSDView *view;
@property (nonatomic) OSDRenderer* osdRenderer;


@end
