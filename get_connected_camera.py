import cv2

def get_camera_info(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    
    # Get the camera details
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Release the camera
    cap.release()
    
    return {
        'Index': index,
        'Width': width,
        'Height': height,
        'FPS': fps
    }

def get_all_camera_info(camera_indices):
    camera_info = []
    for index in camera_indices:
        info = get_camera_info(index)
        if info is not None:
            camera_info.append(info)
    
    return camera_info

# Specify the camera indices
camera_indices = [0, 1, 2, 3, 4, 5]

# Get the camera details
camera_info = get_all_camera_info(camera_indices)

# Print the camera details
for info in camera_info:
    print(f"Camera Index: {info['Index']}")
    print(f"Resolution: {info['Width']}x{info['Height']}")
    print(f"FPS: {info['FPS']}")
    print()
