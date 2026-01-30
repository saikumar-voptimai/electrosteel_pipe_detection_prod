import cv2

def resize_for_inference(frame, target_width=960):
    """
    Resize an image frame to have the target width, maintaining aspect ratio.
    If the frame width is already less than or equal to target_width, returns the original frame
    """
    h, w = frame.shape[:2]                  # Ex: VA - Imaging --> (w2620, h1216)    
    if w <= target_width:                   # 1216 < 960? false
        return frame
    scale = target_width / float(w)         # scale = 960 / 2620 = 0.366
    new_h = int(h * scale)                  # new_h = 1216 * 0.366 = 445 - So that aspect ratio is maintained
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_LINEAR) # Resize to (w960, h445)