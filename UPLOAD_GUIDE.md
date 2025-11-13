# Image Upload Guide for PatchCore Training Web App

## Overview

The web app now includes a comprehensive image upload feature that allows you to create and manage training datasets directly through the browser interface. You no longer need to manually organize files on disk before training.

## Features

✅ **Create Datasets**: Create new datasets with proper folder structure
✅ **Upload Normal Images**: Upload images of normal/good products
✅ **Upload Abnormal Images**: Upload images of defective/anomalous products
✅ **Image Preview**: View uploaded images with thumbnails
✅ **Delete Images**: Remove individual images from the dataset
✅ **Statistics**: Real-time count of uploaded images
✅ **Quick Actions**: Navigate directly to config generation and training
✅ **Multi-language**: Full support for English and Japanese

## How to Use

### Step 1: Start the Web App

On Windows, run:
```bash
python app.py
```

The app will start at `http://localhost:5000`

### Step 2: Navigate to Upload Page

Click on **"Upload"** (アップロード) in the navigation menu.

### Step 3: Create a New Dataset

1. Click the **"➕ Create New"** button
2. Enter a dataset name (e.g., `my_product_dataset`)
   - Use only lowercase letters, numbers, and underscores
   - Example: `circuit_board_v1`, `bottle_caps_2024`
3. Click **"Create"**

The app will automatically create the following folder structure:
```
datasets/
└── my_product_dataset/
    ├── normal/      (for normal/good images)
    └── abnormal/    (for defective images)
```

### Step 4: Upload Normal Images

1. In the **"✅ Upload Normal Images"** section:
   - Click **"Select Images"** to choose files
   - Select multiple images (JPG, PNG, BMP, TIFF supported)
   - Click **"⬆️ Upload Normal Images"**

The images will be uploaded and displayed as thumbnails below.

### Step 5: Upload Abnormal Images

1. In the **"❌ Upload Abnormal Images"** section:
   - Click **"Select Images"** to choose files
   - Select images showing defects or anomalies
   - Click **"⬆️ Upload Abnormal Images"**

**Note**: Abnormal images are optional. You can train with only normal images for pure anomaly detection.

### Step 6: Review Your Dataset

The statistics panel at the top shows:
- **Normal Images**: Count of normal images uploaded
- **Abnormal Images**: Count of abnormal images uploaded
- **Total Images**: Combined total
- **Dataset Path**: Full path on the server

### Step 7: Generate Configuration

Once you've uploaded your images:

1. Click **"➡️ Generate Config for This Dataset"** in the Quick Actions section
2. The config page will open with the dataset path pre-filled
3. Adjust training parameters as needed:
   - Image size (default: 256x256)
   - Training epochs (default: 1)
   - Batch size, learning rate, etc.
4. Click **"Generate Configuration"**
5. Save the configuration file

### Step 8: Train the Model

1. Click **"🚀 Go to Training"** from the upload page, or navigate to the **Train** page
2. Select your generated configuration from the dropdown
3. Click **"Start Training"**
4. Monitor the training progress in real-time

### Step 9: Test with Inference

After training completes:
1. Navigate to the **Inference** page
2. Select your trained model
3. Upload test images to check for anomalies
4. View anomaly scores and heatmaps

## Managing Datasets

### View Uploaded Images

All uploaded images are displayed as thumbnails with:
- Image preview
- Filename
- Delete button (appears on hover)

### Delete Individual Images

1. Hover over an image thumbnail
2. Click the **"×"** button that appears
3. Confirm the deletion

### Delete Entire Dataset

To remove a complete dataset:
1. Select the dataset from the dropdown
2. Scroll to **"Quick Actions"**
3. Click **"🗑️ Delete This Dataset"**
4. Confirm the deletion (⚠️ This cannot be undone!)

### Switch Between Datasets

Use the dataset dropdown at the top to switch between different datasets you've created.

## Dataset Storage Location

All uploaded datasets are stored in:
```
datasets/
├── dataset1/
│   ├── normal/
│   └── abnormal/
├── dataset2/
│   ├── normal/
│   └── abnormal/
└── ...
```

On Windows, this will typically be:
```
C:\Users\YourUsername\train_patchcore_webapp\datasets\
```

## Supported Image Formats

- **JPG / JPEG** (.jpg, .jpeg)
- **PNG** (.png)
- **BMP** (.bmp)
- **TIFF** (.tif, .tiff)

## Tips and Best Practices

### 1. Organize Your Images Before Upload

Before starting:
- Review your images and separate normal from abnormal
- Remove duplicates or corrupted files
- Ensure images show the same type of product/object

### 2. Minimum Dataset Size

For effective training:
- **Minimum**: 20-30 normal images
- **Recommended**: 50+ normal images
- **Abnormal images**: Optional, but 10-20 help with validation

### 3. Image Quality

- Use consistent lighting and angles
- Ensure images are in focus
- Keep background similar across images
- Higher resolution is better (will be resized during training)

### 4. Naming Convention

The app handles file naming automatically, including:
- Duplicate filename detection
- Automatic timestamp addition for conflicts
- Secure filename sanitization

### 5. Upload in Batches

For large datasets:
- Upload images in smaller batches (e.g., 50-100 at a time)
- Monitor the upload status messages
- Check statistics after each batch

### 6. Backup Your Datasets

Important datasets should be backed up:
- The `datasets/` folder contains all uploaded images
- Copy this folder to an external drive or cloud storage
- You can also download trained models from the Train page

## Troubleshooting

### Upload Fails

**Issue**: Images don't upload
**Solution**:
- Check file format (only image files supported)
- Ensure file names don't have special characters
- Try smaller batches if uploading many files
- Check available disk space

### Dataset Not Found

**Issue**: Can't see your dataset
**Solution**:
- Click the **"🔄 Refresh"** button
- Check if you selected the dataset from the dropdown
- Verify the dataset was created successfully

### Images Not Displaying

**Issue**: Thumbnails show broken images
**Solution**:
- Refresh the page
- Check browser console for errors (F12)
- Verify images were uploaded successfully (check statistics)

### Config Generation Fails

**Issue**: Can't generate config for uploaded dataset
**Solution**:
- Ensure you have at least 1 normal image uploaded
- Check that the dataset path is valid
- Verify folder permissions on Windows

### Training Won't Start

**Issue**: Training fails with dataset error
**Solution**:
- Verify dataset structure: `normal/` and `abnormal/` folders exist
- Check that you generated a config file first
- Ensure at least 1 image exists in the `normal/` folder

## Advanced Usage

### Using External Dataset Tools

You can also:
1. Create datasets using the web interface
2. Add more images directly to the `datasets/your_dataset/normal/` folder on disk
3. Refresh the upload page to see the updated count

### Multiple Training Runs

For the same dataset:
1. Keep the dataset in the Upload page
2. Generate multiple configs with different parameters
3. Train separate models to compare results

### Dataset Versioning

Create multiple versions:
- `product_v1`, `product_v2`, etc.
- Helps track improvements over time
- Easy to compare training results

## Windows-Specific Notes

### File Paths

On Windows, paths use backslashes:
```
C:\Users\YourName\train_patchcore_webapp\datasets\my_dataset
```

The app handles this automatically.

### Running the Server

Always run from Command Prompt or PowerShell:
```bash
cd C:\path\to\train_patchcore_webapp
python app.py
```

### Firewall

If you can't access the app:
1. Check Windows Firewall settings
2. Allow Python through the firewall
3. Ensure port 5000 is not blocked

### Large Uploads

For very large images or many files:
- Windows may show "Not Responding" during upload
- This is normal - wait for the upload to complete
- The progress will update once processing finishes

## Summary Workflow

```
1. Upload Page → Create Dataset
2. Upload Page → Upload Normal Images
3. Upload Page → Upload Abnormal Images (optional)
4. Upload Page → "Generate Config" button
5. Config Page → Adjust parameters → Generate
6. Train Page → Select config → Start Training
7. Inference Page → Test trained model
```

## Need Help?

If you encounter issues:
1. Check this guide for troubleshooting steps
2. Review the browser console (F12) for errors
3. Check the terminal where `app.py` is running for server logs
4. Verify your dataset structure matches the expected format

## Feature Comparison

### Before (Manual)
- ❌ Manually create folders on disk
- ❌ Copy/paste images into folders
- ❌ Navigate file system
- ❌ Remember folder structure requirements
- ❌ Update config with complex paths

### Now (Web Upload)
- ✅ Create datasets in the browser
- ✅ Drag and drop or select images
- ✅ Visual preview of uploaded images
- ✅ Automatic folder structure
- ✅ One-click config generation
- ✅ Real-time statistics
- ✅ Manage all datasets from one page

---

**Enjoy your streamlined anomaly detection workflow!** 🚀
