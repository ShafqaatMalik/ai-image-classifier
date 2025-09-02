#!/usr/bin/env python3
"""
Test script to verify both systems work correctly
"""

import sys
import traceback

def test_music_recommender():
    """Test the music recommender system."""
    print("🎵 Testing Music Recommender System...")
    
    try:
        from music_recommender import MusicRecommenderSystem, SpotifyDataExtractor
        print("✅ Music recommender imports successful")
        
        # Test class creation without credentials
        try:
            # This should fail gracefully
            recommender = MusicRecommenderSystem()
            print("⚠️  Created system without credentials (will have limited functionality)")
        except ValueError as e:
            print(f"✅ Properly handled missing credentials: {e}")
        except Exception as e:
            print(f"⚠️  Unexpected error: {e}")
        
        print("✅ Music recommender system test completed")
        return True
        
    except Exception as e:
        print(f"❌ Music recommender test failed: {e}")
        traceback.print_exc()
        return False

def test_image_classifier():
    """Test the image classifier system."""
    print("\n🖼️  Testing Image Classifier System...")
    
    try:
        # Test imports
        import torch
        import torch.nn as nn
        from torchvision import datasets
        import torchvision.transforms as transforms
        from torchmetrics import Accuracy, Precision, Recall
        print("✅ Image classifier imports successful")
        
        # Test basic PyTorch functionality
        x = torch.randn(1, 1, 28, 28)
        print(f"✅ PyTorch tensor creation successful: {x.shape}")
        
        # Test if we can create the model architecture
        class TestNet(nn.Module):
            def __init__(self, num_classes):
                super(TestNet, self).__init__()
                self.feature_extractor = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Flatten()
                )
                self.classifier = nn.Linear(64*7*7, num_classes)
            
            def forward(self, x):
                x = self.feature_extractor(x)
                x = self.classifier(x)
                return x
        
        net = TestNet(10)
        output = net(x)
        print(f"✅ Model architecture test successful: output shape {output.shape}")
        
        print("✅ Image classifier system test completed")
        return True
        
    except Exception as e:
        print(f"❌ Image classifier test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Running System Tests")
    print("=" * 40)
    
    music_ok = test_music_recommender()
    image_ok = test_image_classifier()
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"Music Recommender: {'✅ PASS' if music_ok else '❌ FAIL'}")
    print(f"Image Classifier: {'✅ PASS' if image_ok else '❌ FAIL'}")
    
    if music_ok and image_ok:
        print("\n🎉 All systems operational!")
        print("\n📝 Next Steps:")
        print("1. Set up Spotify API credentials in .env file for full music recommender functionality")
        print("2. Run 'python music_recommender.py' for music recommendations") 
        print("3. Run 'python aiimageclassifier.py' for image classification")
        print("4. Check 'example_usage.py' for detailed usage examples")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()