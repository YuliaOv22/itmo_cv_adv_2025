from pathlib import Path
from utils.read_save_files import read_images_from_directory, save_paths_to_file
import scripts.detect_with_orb_method as om
from scripts.detect_with_cnn import TransformerEmbedder
import time
from utils.log import TeeLoggerContext

if __name__ == "__main__":

    log_path = Path("logs/predict_on_data.log")
    log_path.parent.mkdir(exist_ok=True)

    with TeeLoggerContext(log_path):

        start_time = time.time()
        print("-------------------Private data-------------------")
        # Настройка переменных
        private_data_dir = Path("data")
        train_images_path_private = f"{private_data_dir}/train"
        test_images_path_private = f"{private_data_dir}/test"
        output_dir_path = Path("output")
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        is_resnet = True

        # Чтение изображений из папок
        train_images_private, test_images_private = read_images_from_directory(
            train_images_path_private, test_images_path_private, image_extensions
        )

        output_path_orb_csv = output_dir_path / "pred_orb_submission.csv"
        output_path_orb_resnet_csv = output_dir_path / "pred_orb_resnet_submission.csv"
        output_path_orb_txt = output_dir_path / "pred_orb_final.txt"
        output_path_orb_resnet_txt = output_dir_path /"pred_orb_resnet_final.txt"

        # Метод ORB (Oriented FAST and Rotated BRIEF)
        print("-------------------Метод ORB-------------------")

        # # Параметры
        IMAGE_SIZE = (128, 128)

        # Предзагрузка изображений
        print("Загрузка тренировочных изображений...")
        train_images_data = om.load_images(train_images_private, target_size=IMAGE_SIZE)

        print("Загрузка тестовых изображений...")
        test_images_data = om.load_images(test_images_private, target_size=IMAGE_SIZE)

        orb_preds = om.get_image_comparison(
            train_images=train_images_data,
            test_images=test_images_data,
            test_paths=test_images_private,
            threshold=40,
        )

        preds_labels = [1 if img in orb_preds else 0 for img in test_images_private]
        test_images_private_ = [img.name for img in test_images_private]
        save_paths_to_file(
            output_path_orb_csv, [test_images_private_, preds_labels], format="csv"
        )
        save_paths_to_file(output_path_orb_txt, orb_preds, format="txt")

        end_time = time.time()
        final_time = time.strftime("%M:%S", time.gmtime(end_time - start_time))
        print(f"Итоговое время выполнения ORB: {final_time}")


        if is_resnet:
            print("-------------------ResNet50-------------------")
            left_images = [img for img in test_images_private if img not in orb_preds]
            IMAGE_SIZE = (512, 512)
            embedder = TransformerEmbedder(IMAGE_SIZE, model_name="resnet50", device="cuda")

            # Создание эмбеддингов
            print("Создание эмбеддингов для тренировочных изображений...")
            train_embeddings = embedder.compute_embeddings(train_images_private)

            print("Создание эмбеддингов для тестовых изображений...")
            test_embeddings = embedder.compute_embeddings(left_images)

            resnet_preds = embedder.get_image_comparison(
                train_embeddings=train_embeddings,
                test_embeddings=test_embeddings,
                test_paths=left_images,
                threshold=0.85,
            )

            preds = orb_preds + resnet_preds

            preds_labels = [1 if img in preds else 0 for img in test_images_private]
            assert len(preds_labels) == len(
                test_images_private
            ), "Количество предсказаний не совпадает с количеством тестовых изображений"

            test_images_private = [img.name for img in test_images_private]
            save_paths_to_file(
                output_path_orb_resnet_csv, [test_images_private, preds_labels], format="csv"
            )
            save_paths_to_file(output_path_orb_resnet_txt, preds, format="txt")

            end_time = time.time()
            final_time = time.strftime("%M:%S", time.gmtime(end_time - start_time))
            print(f"Итоговое время выполнения метода ORB + ResNet50: {final_time}")
