import cv2
import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput


DEFAULT_YOLO_INPUT_SIZE = (640, 640)
DEFAULT_IMAGE_INPUT_SIZE = (640, 360)
YOLO_KEYPOINTS_SHAPE = [17, 3]
YOLO_INPUT_NAME = "images"
YOLO_OUTPUT_NAME = "output0"
YOLO_PRECISION = "FP16"


def scale_boxes(img1_shape: tuple, boxes: np.ndarray, img0_shape: tuple, ratio_pad: tuple = None, padding: bool = True,
                xywh: bool = False):
    """
    Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
    specified in (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        xywh (bool): The box format is xywh or not, default=False.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def clip_boxes(boxes: np.ndarray, shape: tuple):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.
    """
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def scale_coords(img1_shape: tuple, coords: np.ndarray, img0_shape: tuple, ratio_pad: tuple = None,
                 normalize: bool = False, padding: bool = True):
    """
    Rescale segment coordinates (xy) from img1_shape to img0_shape.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords


def clip_coords(coords: np.ndarray, shape: tuple):
    """
    Clip line coordinates to the image boundaries.
    """
    coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
    coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    return coords


def xywh2xyxy(x: np.ndarray):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def xyxy2xywh(x: np.ndarray):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x, y) is the
    top-left corner and (width, height) is the bottom-right corner. Note: ops per 2 channels faster than per channel.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1]
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def empty_like(x: np.ndarray):
    """Creates empty torch.Tensor or np.ndarray with same shape as input and float32 dtype."""
    return np.empty_like(x, dtype=np.float32)


def preprocess_image(image: np.ndarray, yolo_size=DEFAULT_YOLO_INPUT_SIZE, input_size=DEFAULT_IMAGE_INPUT_SIZE):
    assert input_size[0] <= yolo_size[0]
    assert input_size[1] <= yolo_size[1]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_size)

    _image = 128 * np.ones((yolo_size[1], yolo_size[0], 3), dtype=np.float32)
    _image[:input_size[1], :input_size[0], :] = image
    image_array = _image / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))  # CHW
    image_array = np.expand_dims(image_array, axis=0)  # Batch size
    return image_array.astype(np.float16)  # Convert to FP16


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
    """Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Args:
        boxes (np.ndarray): Array of shape (n, 4) where each row is [x1, y1, x2, y2].
        scores (np.ndarray): Array of shape (n,) containing scores for each box.
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        np.ndarray: Array of indices of boxes to keep.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # Sort by scores in descending order

    keep = []
    while order.size > 0:
        i = order[0]  # Index of the current highest score
        keep.append(i)

        # Compute IoU of the kept box with the remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Suppress boxes with IoU above the threshold
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return np.array(keep)


def postprocess_prediction(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=1,
        max_nms=30000,
        max_wh=7680,
        in_place=True
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (np.ndarray): A numpy array of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
        agnostic (bool): If True, the model is agnostic to the number of classes.
        multi_label (bool): If True, each box may have multiple labels.
        labels (list): A list of lists containing apriori labels for each image.
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int): The number of classes output by the model.
        max_nms (int): The maximum number of boxes into NMS.
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction array will be modified in place.

    Returns:
        list[np.ndarray]: A list of length batch_size, where each element is an array of
            shape (num_boxes, 6 + num_masks) containing the kept boxes.
    """

    if not prediction.flags.writeable:
        prediction = prediction.copy()

    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    if prediction.shape[-1] == 6:  # end-to-end model
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        return output

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].max(axis=1) > conf_thres  # candidates

    # Settings
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    prediction = np.transpose(prediction, (0, 2, 1))  # shape(1, 84, 6300) -> shape(1, 6300, 84)

    if in_place:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
    else:
        prediction = np.concatenate((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), axis=-1)

    output = [np.zeros((0, 6 + nm))] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # confidence

        if labels and len(labels[xi]):
            lb = np.array(labels[xi])
            v = np.zeros((len(lb), nc + nm + 4))
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[np.arange(len(lb)), lb[:, 0].astype(int) + 4] = 1.0  # cls
            x = np.concatenate((x, v), axis=0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        box, cls, mask = np.split(x, [4, 4 + nc], axis=1)

        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None].astype(float), mask[i]), axis=1)
        else:  # best class only
            conf = cls.max(axis=1, keepdims=True)
            j = cls.argmax(axis=1, keepdims=True)
            x = np.concatenate((box, conf, j.astype(float), mask), axis=1)[conf.ravel() > conf_thres]

        # Check shape
        if not x.shape[0]:  # no boxes
            continue

        if x.shape[0] > max_nms:  # excess boxes
            x = x[np.argsort(-x[:, 4])[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        boxes = x[:, :4] + c  # boxes (offset by class)
        keep = nms(boxes, scores, iou_thres)
        keep = keep[:max_det]  # limit detections

        output[xi] = x[keep]

    return output


class Boxes:
    def __init__(self, boxes: np.ndarray, img_shape: tuple):
        self.boxes = boxes
        self.img_shape = img_shape

    @property
    def xyxy(self) -> np.ndarray:
        return self.boxes.astype("int")

    @property
    def xyxyn(self):
        xyxy = np.copy(self.boxes)
        xyxy[..., [0, 2]] /= self.img_shape[0]
        xyxy[..., [1, 3]] /= self.img_shape[1]
        return xyxy

    @property
    def xywh(self) -> np.ndarray:
        return xyxy2xywh(self.xyxy)

    @property
    def xywhn(self) -> np.ndarray:
        return xyxy2xywh(self.xyxyn)


class Keypoints:
    def __init__(self, keypoints: np.ndarray, img_shape: tuple):
        self.keypoints = keypoints
        self.img_shape = img_shape

    @property
    def xy(self) -> np.ndarray:
        return self.keypoints.astype("int")

    @property
    def xyn(self) -> np.ndarray:
        xy = np.copy(self.keypoints)
        xy[..., 0] /= self.img_shape[0]
        xy[..., 1] /= self.img_shape[1]
        return xy


class Prediction:
    def __init__(self, boxes: np.ndarray, keypoints: np.ndarray, img_shape: tuple):
        self._boxes = Boxes(boxes[:, :4], img_shape)
        self._scores = boxes[:, 4]
        self._keypoints = Keypoints(keypoints[:, :, :2] if len(keypoints) else np.empty((0, *YOLO_KEYPOINTS_SHAPE)),
                                    img_shape)
        self._keypoints_scores = keypoints[:, :, 2] if len(keypoints) else np.empty(0)

    @property
    def boxes(self) -> Boxes:
        return self._boxes

    @property
    def boxes_scores(self) -> np.ndarray:
        return self._scores

    @property
    def keypoints(self) -> Keypoints:
        return self._keypoints

    @property
    def keypoints_scores(self) -> np.ndarray:
        return self._keypoints_scores


class YoloTritonClient:
    def __init__(self, triton_server_url: str, model_name: str, model_version: str):
        self.client = InferenceServerClient(url=triton_server_url)
        self.model_name = model_name
        self.model_version = model_version

    def inference(self, image: np.ndarray):
        image_array = preprocess_image(image)

        # Prepare input tensor
        inputs = [
            InferInput(YOLO_INPUT_NAME, image_array.shape, YOLO_PRECISION)
        ]
        inputs[0].set_data_from_numpy(image_array)

        # Prepare output tensor
        outputs = [
            InferRequestedOutput(YOLO_OUTPUT_NAME)
        ]

        # Perform inference
        response = self.client.infer(model_name=self.model_name, model_version=self.model_version, inputs=inputs,
                                     outputs=outputs)

        # Extract and process the output
        predictions = response.as_numpy(YOLO_OUTPUT_NAME)
        predictions = postprocess_prediction(predictions)

        kpt_shape = YOLO_KEYPOINTS_SHAPE
        img_shape = DEFAULT_YOLO_INPUT_SIZE

        pred = predictions[0]  # just 1-batch inference

        pred[:, :4] = scale_boxes(img_shape, pred[:, :4], img_shape).round()
        pred_kpts = pred[:, 6:].reshape(len(pred), *kpt_shape) if len(pred) else pred[:, 6:]
        pred_kpts = scale_coords(img_shape, pred_kpts, img_shape)

        raw_boxes = pred[:, :5]
        raw_keypoints = pred_kpts

        result = Prediction(raw_boxes, raw_keypoints, DEFAULT_IMAGE_INPUT_SIZE)

        return result



if __name__ == "__main__":

    TRITON_SERVER_URL = "192.168.2.135:8001"

    def test_unary():

        import tracemalloc

        tracemalloc.start()

        yolo = YoloTritonClient(
            triton_server_url=TRITON_SERVER_URL,
            model_name="yolo",
            model_version="1"
        )

        image = cv2.imread("data/test.jpeg")

        result = yolo.inference(image)

        for rect in result.boxes.xyxy:
            cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (0, 255, 0), 2)

        for kpts in result.keypoints.xyxy:
            for kpt in kpts:
                cv2.circle(image, (int(kpt[0]), int(kpt[1])), 3, (0, 0, 255), -1)

        cv2.imwrite("data/out.jpeg", image)

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        for stat in top_stats[:3]:
            print(stat)

        tracemalloc.stop()

    test_unary()