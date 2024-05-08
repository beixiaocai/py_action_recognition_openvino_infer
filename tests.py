from controllers.action_recognition_algorithm import *

if __name__ == '__main__':

    sample_duration = frames2decode  # Decoder input size - From Cell 5_7

    encoder_output = []
    top_k = 5
    decoded_labels = []
    decoded_top_probs = []

    text_inference_template = "Infer Time:{Time:.1f}ms,{fps:.1f}FPS"
    text_template = "{label},{conf:.2f}%"
    parent_dir = os.path.dirname(CURRENT_DIR)
    url = "data/kd.mp4"
    # url = 0
    print("url=", url)

    cap = cv2.VideoCapture(url)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while True:
        r, frame = cap.read()
        if r:
            # frame = cv2.imread("test_images/image2117.jpg")

            scale = 1280 / max(frame.shape)

            # Adaptative resize for visualization.
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                frame = cv2.resize(frame, (1280, 720))

            # Preprocess frame before Encoder.
            (preprocessed, _) = preprocessing(frame, height_en)

            # Measure processing time.
            start_time = time.time()

            # Encoder Inference per frame
            output_key_en = compiled_model_en.output(0)
            infer_result_encoder = compiled_model_en([preprocessed])[output_key_en]
            encoder_output.append(infer_result_encoder)

            # print(len(encoder_output))
            # print(sample_duration)

            if len(encoder_output) == sample_duration:
                # print("encode success")
                # Concatenate sample_duration frames in just one array
                decoder_input = np.concatenate(encoder_output, axis=0)
                # Organize input shape vector to the Decoder (shape: [1x16x512]]
                decoder_input = decoder_input.transpose((2, 0, 1, 3))
                decoder_input = np.squeeze(decoder_input, axis=3)
                output_key_de = compiled_model_de.output(0)
                # Get results on action-recognition-0001-decoder model
                result_de = compiled_model_de([decoder_input])[output_key_de]

                # 规范化logits以获得沿指定轴的置信值
                probs = softmax(result_de - np.max(result_de))

                # 将最高概率解码为相应的标签名称 start
                top_ind = np.argsort(-1 * probs)[:top_k]
                out_label = np.array(labels)[top_ind.astype(int)]

                # decoded_labels = [out_label[0][0], out_label[0][1], out_label[0][2]]
                decoded_labels = []
                for i in range(top_k):
                    decoded_labels.append(out_label[0][i])

                top_probs = np.array(probs)[0][top_ind.astype(int)]

                # decoded_top_probs = [top_probs[0][0], top_probs[0][1], top_probs[0][2]]
                decoded_top_probs = []
                for i in range(top_k):
                    decoded_top_probs.append(top_probs[0][i])

                # 将最高概率解码为相应的标签名称 end

                encoder_output = []

            # print("显示检测结果")
            # print(decoded_labels)
            # print(decoded_top_probs)

            for i in range(len(decoded_labels)):
                display_text = text_template.format(
                    label=decoded_labels[i],
                    conf=decoded_top_probs[i] * 100,
                )
                display_text_fnc(frame, display_text, i)
            display_text = text_inference_template.format(Time=0, fps=fps)
            display_text_fnc(frame, display_text, top_k)

            cv2.imshow("frame", frame)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
        else:
            break
