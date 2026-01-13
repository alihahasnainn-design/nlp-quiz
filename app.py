import gradio as gr
import cv2
import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
import requests
from io import BytesIO
import time

# YOLOv10 class for inference
class YOLOv10:
    def __init__(self, path):
        self.session = onnxruntime.InferenceSession(
            path, providers=onnxruntime.get_available_providers()
        )
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image, conf_threshold=0.3):
        input_tensor = self.prepare_input(image)
        boxes, scores, class_ids = self.inference(input_tensor, conf_threshold)
        self.draw_detections(image, boxes, scores, class_ids)
        return image

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def inference(self, input_tensor, conf_threshold=0.3):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        boxes, scores, class_ids = self.process_output(outputs, conf_threshold)
        return boxes, scores, class_ids

    def process_output(self, output, conf_threshold=0.3):
        predictions = np.squeeze(output[0])
        scores = predictions[:, 4]
        predictions = predictions[scores > conf_threshold, :]
        scores = scores[scores > conf_threshold]
        if len(scores) == 0:
            return [], [], []
        class_ids = predictions[:, 5].astype(int)
        boxes = self.extract_boxes(predictions)
        # Apply NMS
        keep = multiclass_nms(boxes, scores, class_ids, iou_threshold=0.5)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        return boxes, scores, class_ids

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = xywh2xyxy(boxes)  # Convert xywh to xyxy
        return boxes

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, boxes, scores, class_ids, mask_alpha=0.3):
        draw_detections(image, boxes, scores, class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

# Utils (class names, NMS, drawing functions)
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

def nms(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]
    return keep_boxes

def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)
    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices,:]
        class_scores = scores[class_indices]
        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])
    return keep_boxes

def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
    iou = intersection_area / union_area
    return iou

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]
        draw_box(image, box, color)
        label = class_names[class_id]
        caption = f"{label} {int(score * 100)}%"
        draw_text(image, caption, box, color, font_size, text_thickness)

def draw_box(image, box, color=(0, 0, 255), thickness=2):
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

def draw_text(image, text, box, color=(0, 0, 255), font_size=0.001, text_thickness=2):
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, text_thickness)
    th = int(th * 1.2)
    cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)
    cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

# Load model
model_file = hf_hub_download(repo_id="onnx-community/yolov10n", filename="onnx/model.onnx")
model = YOLOv10(model_file)

# Detection function (for all inputs)
def detection(image, conf_threshold=0.3):
    if image is None:
        return None
    image = image.copy()  # Avoid modifying original
    return model.detect_objects(image, conf_threshold)

# Load image from URL
def load_image_from_url(url):
    response = requests.get(url)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

# Gradio app
css = """
.my-group {max-width: 600px !important; max-height: 600px !important;}
.my-column {display: flex !important; justify-content: center !important; align-items: center !important;}
"""

rtc_configuration = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

with gr.Blocks(css=css) as demo:
    gr.HTML("<h1 style='text-align: center'>Object Detection App (YOLOv10n from Hugging Face)</h1>")
    
    with gr.Tab("Real-Time Webcam"):
        with gr.Column(elem_classes=["my-column"]):
            with gr.Group(elem_classes=["my-group"]):
                webcam_in = gr.Image(label="Webcam Stream", source="webcam", streaming=True)
                conf_webcam = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.3)
                webcam_in.stream(fn=detection, inputs=[webcam_in, conf_webcam], outputs=[webcam_in], time_limit=10)
    
    with gr.Tab("Upload Image"):
        with gr.Row():
            img_in = gr.Image(label="Upload Image")
            conf_upload = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.3)
        output_upload = gr.Image(label="Detected Objects")
        btn_upload = gr.Button("Detect")
        btn_upload.click(detection, inputs=[img_in, conf_upload], outputs=output_upload)
    
    with gr.Tab("Image from URL"):
        with gr.Row():
            url_in = gr.Textbox(label="Enter Image URL")
            conf_url = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.3)
        output_url = gr.Image(label="Detected Objects")
        btn_url = gr.Button("Detect")
        btn_url.click(lambda url, conf: detection(load_image_from_url(url), conf), inputs=[url_in, conf_url], outputs=output_url)

if __name__ == "__main__":
    demo.launch()
