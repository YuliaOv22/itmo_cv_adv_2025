from pathlib import Path
from utils.read_save_files import read_images_from_directory, save_paths_to_file
import scripts.orb_method as om
from scripts.cnn import TransformerEmbedder
import time
from scripts.log import TeeLoggerContext

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
        output_dir_path = "output"
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        method = "orb_resnet"

        # Чтение изображений из папок
        train_images_private, test_images_private = read_images_from_directory(
            train_images_path_private, test_images_path_private, image_extensions
        )

        output_path_csv = Path(
            f"{output_dir_path}/pred_{method}_final_submission.csv"
        )
        output_path_txt = Path(f"{output_dir_path}/pred_{method}_final.txt")

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
            output_path_csv, [test_images_private, preds_labels], format="csv"
        )
        save_paths_to_file(output_path_txt, preds, format="txt")

        end_time = time.time()
        final_time = time.strftime("%M:%S", time.gmtime(end_time - start_time))
        print(f"Итоговое время выполнения: {final_time}")
