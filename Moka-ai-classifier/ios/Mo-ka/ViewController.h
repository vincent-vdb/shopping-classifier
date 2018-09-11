//
//  ViewController.h
//  Mo-ka
//
//  Created by Valentin Dijkstra on 06/10/2017.
//  Copyright © 2017 Valentin Dijkstra. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <AVFoundation/AVCaptureSession.h>
#import <AVFoundation/AVCaptureVideoPreviewLayer.h>
#import <AVFoundation/AVCaptureStillImageOutput.h>
#import <AVFoundation/AVCaptureInput.h>

@interface ViewController : UIViewController <UIImagePickerControllerDelegate, UINavigationControllerDelegate>
@property (strong, nonatomic) IBOutlet UIImageView *imageView;
@property (nonatomic, retain) IBOutlet UIView *vImagePreview;
@property (nonatomic, retain) AVCaptureStillImageOutput *stillImageOutput;
@property (weak, nonatomic) IBOutlet UILabel *valueLabel;
@property (weak, nonatomic) IBOutlet UIStepper *stepper;
@property (weak, nonatomic) NSTimer *sendTimer;

- (IBAction)takePhoto:(UIButton *)sender;
- (IBAction)stepperValueChanged:(UIStepper *)sender;


@end

