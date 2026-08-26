"""
Helper script to run N2V in a separate TensorFlow environment.
Called via subprocess from the main evaluation script.
"""

import sys
import numpy as np
import skimage.io as io
from pathlib import Path

def run_n2v_on_image(input_path, output_path, epochs=250):
    """Run N2V denoising on a single image."""
    from n2v.models import N2VConfig, N2V
    from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    import tempfile
    import shutil
    import gc

    # Load image
    img = io.imread(input_path)
    if img.max() > 1:
        img = img.astype(np.float32) / 255.0
    
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    
    # N2V processing
    img_4d = img[np.newaxis, ...]
    datagen = N2V_DataGenerator()
    X = datagen.generate_patches_from_list([img_4d], shape=(64, 64))
    X_train, X_val = train_test_split(X, test_size=0.20, random_state=42, shuffle=True)
    
    config = N2VConfig(
        X, unet_kern_size=3, unet_n_first=64, unet_n_depth=2,
        train_steps_per_epoch=int(X.shape[0] / 128), train_epochs=epochs,
        train_loss='mse', batch_norm=True, train_batch_size=16,
        n2v_perc_pix=0.198, n2v_patch_shape=(64, 64),
        n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5,
        single_net_per_channel=False
    )
    
    temp_dir = tempfile.mkdtemp()
    model = N2V(config, name="n2v_temp", basedir=temp_dir)
    model.train(X_train, X_val)
    denoised = model.predict(img, axes="YXC", n_tiles=(2, 2, 1))
    
    # Cleanup
    del model, X, X_train, X_val
    tf.keras.backend.clear_session()
    gc.collect()
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Save result
    denoised_uint8 = (np.clip(denoised, 0, 1) * 255).astype(np.uint8)
    io.imsave(output_path, denoised_uint8)
    
    return 0


if __name__ == "__main__":
    # Usage: python run_n2v_external.py input.png output.png [epochs]
    if len(sys.argv) < 3:
        print("Usage: python run_n2v_external.py <input.png> <output.png> [epochs]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 250
    
    try:
        run_n2v_on_image(input_path, output_path, epochs)
        sys.exit(0)
    except Exception as e:
        print(f"N2V Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)