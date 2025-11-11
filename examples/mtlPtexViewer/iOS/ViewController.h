//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#import <UIKit/UIKit.h>
#import <MetalKit/MetalKit.h>
#import "../mtlPtexViewer.h"

@interface ViewController : UIViewController<
                                MTKViewDelegate,
                                UIPickerViewDelegate,
                                UIPickerViewDataSource,
                                OSDRendererDelegate
                            >

@property (weak, nonatomic) IBOutlet UILabel *refLvlLabel;
@property (weak, nonatomic) IBOutlet UILabel *tesLvLlabel;
@property (weak, nonatomic) IBOutlet UILabel *frameTimeLabel;
@property (weak, nonatomic) IBOutlet UIPickerView *colorModePickerView;
@property (weak, nonatomic) IBOutlet UIPickerView *normalModePickerView;
@property (weak, nonatomic) IBOutlet UIPickerView *displacementModePickerView;
@property (weak, nonatomic) IBOutlet UIStepper *tessellationStepper;
@property (weak, nonatomic) IBOutlet UIStepper *refinementStepper;
@property (weak, nonatomic) IBOutlet UISwitch *wireframeSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *backpatchCullingSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *backfaceCullingSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *patchClipCullingSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *singleCreaseSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *screenspaceTessellationSwitch;
@property (weak, nonatomic) IBOutlet UISlider *displacementSlider;
@property (weak, nonatomic) IBOutlet UISlider *mipmapBiasSlider;
- (IBAction)stepperChanged:(UIStepper *)sender;
- (IBAction)switchChanged:(UISwitch *)sender;
- (IBAction)sliderChanged:(UISlider*)sender;

@end

