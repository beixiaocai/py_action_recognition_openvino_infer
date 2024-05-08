from controllers.BaseHandler import BaseHandler
import json
import numpy as np
import base64
import cv2
from controllers.action_recognition_algorithm import frames2decode, \
    preprocessing, \
    height_en, \
    compiled_model_en, \
    compiled_model_de, softmax, labels


class IndexHandler(BaseHandler):
    async def get(self, *args, **kwargs):
        data = await self.do()
        self.response_json(data)

    async def do(self):
        res = {
            "code": 1000,
            "msg": "success"
        }
        return res


g_encoder_output = []


class AlgorithmHandler(BaseHandler):
    async def post(self, *args, **kwargs):
        data = await self.do()
        self.response_json(data)

    async def do(self):
        request_params = self.request_post_params()

        # print(request_params.keys())

        happen = False
        happenScore = 0.0
        detects = []

        image_base64 = request_params.get("image_base64", None)  # 接收base64编码的图片并转换成cv2的图片格式
        if image_base64:
            encoded_image_byte = base64.b64decode(image_base64)
            image_array = np.frombuffer(encoded_image_byte, np.uint8)
            # image = turboJpeg.decode(image_array)  # turbojpeg 解码
            image = cv2.imdecode(image_array, cv2.COLOR_RGB2BGR)  # opencv 解码

            frame = image
            image_height = image.shape[0]
            image_width = image.shape[1]

            sample_duration = frames2decode  # Decoder input size - From Cell 5_7

            decoded_labels = []
            decoded_top_probs = []
            top_k = 5
            scale = 1280 / max(frame.shape)

            # Adaptative resize for visualization.
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                frame = cv2.resize(frame, (1280, 720))

            # Preprocess frame before Encoder.
            (preprocessed, _) = preprocessing(frame, height_en)

            # Encoder Inference per frame
            output_key_en = compiled_model_en.output(0)
            infer_result_encoder = compiled_model_en([preprocessed])[output_key_en]
            g_encoder_output.append(infer_result_encoder)

            # print(len(g_encoder_output))
            # print(sample_duration)

            if len(g_encoder_output) == sample_duration:
                # print("encode success")
                # Concatenate sample_duration frames in just one array
                decoder_input = np.concatenate(g_encoder_output, axis=0)
                # Organize input shape vector to the Decoder (shape: [1x16x512]]
                decoder_input = decoder_input.transpose((2, 0, 1, 3))
                decoder_input = np.squeeze(decoder_input, axis=3)
                output_key_de = compiled_model_de.output(0)
                # Get results on action-recognition-0001-decoder model
                result_de = compiled_model_de([decoder_input])[output_key_de]

                # 规范化logits以获得沿指定轴的置信值
                probs = softmax(result_de - np.max(result_de))

                # 将最高概率解码为相应的标签名称 start
                top_idx = np.argsort(-1 * probs)[:top_k]
                out_label = np.array(labels)[top_idx.astype(int)]

                # decoded_labels = [out_label[0][0], out_label[0][1], out_label[0][2]]
                decoded_labels = []
                for i in range(top_k):
                    decoded_labels.append(out_label[0][i])

                top_probs = np.array(probs)[0][top_idx.astype(int)]

                # decoded_top_probs = [top_probs[0][0], top_probs[0][1], top_probs[0][2]]
                decoded_top_probs = []
                for i in range(top_k):
                    decoded_top_probs.append(top_probs[0][i])

                # 将最高概率解码为相应的标签名称 end

                g_encoder_output.clear()

            # print("显示检测结果")
            # print(decoded_labels)
            # print(decoded_top_probs)

            for i in range(len(decoded_labels)):
                score = float(decoded_top_probs[i])
                class_name = decoded_labels[i]

                margin_top = i * 100
                detects.append(
                    {
                        "x1": 100,
                        "y1": 150 + margin_top,
                        "x2": image_width-100,
                        "y2": 350 + margin_top,
                        "class_score": score,
                        "class_name": class_name
                    })

        # detects.append(
        #     {
        #         "x1": 100,
        #         "y1": 200,
        #         "x2": 600,
        #         "y2": 700,
        #         "class_score": 1,
        #         "class_name": "test"
        #     })

        if len(detects) > 0:
            happen = True
            happenScore = 1.0

        res = {
            "code": 1000 if happen else 0,
            "msg": "success",
            "result": {
                "happen": happen,
                "happenScore": happenScore,
                "detects": detects
            }
        }
        return res
