
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageEnhance


# =====================================
# 1. KONFIGURASI APLIKASI
# =====================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "coffee_roast_model.keras"
CLASS_PATH = BASE_DIR / "class_names.json"
DATASET_DIR = BASE_DIR / "test"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CATEGORIES = ["Green", "Light", "Medium", "Dark"]

ROAST_NAMES = {
    "Green": "",
    "Light": "",
    "Medium": "",
    "Dark": "",
}

ROAST_INFO = {
    "Green": {
        "judul": "Biji Kopi Hijau (Belum Disangrai)",
        "deskripsi": (
            "Biji belum melalui proses sangrai. Karakter aroma "
            "dan rasa hasil sangrai belum berkembang."
        ),
    },
    "Light": {
        "judul": "Sangrai Ringan (Light Roast)",
        "deskripsi": (
            "Biji umumnya berwarna cokelat muda. Karakter asal "
            "biji dan keasaman cenderung lebih menonjol."
        ),
    },
    "Medium": {
        "judul": "Sangrai Sedang (Medium Roast)",
        "deskripsi": (
            "Biji umumnya berwarna cokelat sedang. Profil rasa "
            "cenderung seimbang antara karakter asal biji "
            "dan hasil sangrai."
        ),
    },
    "Dark": {
        "judul": "Sangrai Gelap (Dark Roast)",
        "deskripsi": (
            "Biji umumnya berwarna cokelat tua hingga gelap. "
            "Karakter sangrai dapat lebih kuat dengan rasa "
            "pahit atau smoky."
        ),
    },
}

st.set_page_config(
    page_title="Coffee Bean Roast Analyzer",
    page_icon="☕",
    layout="wide",
)


# =====================================
# 2. MEMUAT MODEL AI
# =====================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False,
    )


@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# =====================================
# 3. HEADER
# =====================================
st.title("☕ Coffee Bean Roast Analyzer")

st.write(
    "Aplikasi pengolahan citra digital dan AI untuk "
    "mengatur kontras serta memprediksi tingkat sangrai kopi."
)

st.info(
    "Kategori prediksi: Green, Light, Medium, dan Dark. "
    "Hasil AI merupakan prediksi model, bukan penilaian "
    "menyeluruh terhadap kualitas kopi."
)


# =====================================
# 4. PERIKSA FILE MODEL
# =====================================
if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    st.error(
        "File coffee_roast_model.keras atau class_names.json "
        "tidak ditemukan. Pastikan kedua file berada di folder "
        "yang sama dengan app.py."
    )
    st.stop()

try:
    model = load_model()
    class_names = load_class_names()
except Exception as error:
    st.error(f"Gagal memuat model AI: {error}")
    st.stop()


# =====================================
# 5. PILIH SUMBER GAMBAR
# =====================================
st.subheader("1. Pilih Foto Biji Kopi")

image_source = st.radio(
    "Sumber gambar",
    ["Pilih dari Dataset", "Unggah Gambar Sendiri"],
    horizontal=True,
)

image = None

if image_source == "Pilih dari Dataset":

    st.write("Klik tombol di bawah foto yang ingin dianalisis.")

    if not DATASET_DIR.exists():
        st.error(
            "Folder test tidak ditemukan. Pastikan folder test "
            "berada di folder yang sama dengan app.py."
        )
    else:
        for category in CATEGORIES:
            folder = DATASET_DIR / category

            if not folder.exists():
                st.warning(
                    f"Folder kategori {category} tidak ditemukan."
                )
                continue

            files = sorted(
                [
                    file
                    for file in folder.iterdir()
                    if file.is_file()
                    and file.suffix.lower() in IMAGE_EXTENSIONS
                ],
                key=lambda file: file.name.lower(),
            )[:3]

            if not files:
                continue

            # Judul kategori bisa dihilangkan dengan menghapus
            # baris st.markdown berikut.
            st.markdown(f"#### {ROAST_NAMES[category]}")

            columns = st.columns(3)

            for index, file in enumerate(files):
                with columns[index]:
                    try:
                        with Image.open(file) as source_image:
                            thumbnail = source_image.convert("RGB").copy()

                        # Ukuran thumbnail di galeri dataset.
                        st.image(thumbnail, width=150)

                        st.caption(file.name)

                        image_key = f"{category}_{index}"

                        if st.button(
                            "Pilih foto",
                            key=f"select_{image_key}",
                            use_container_width=True,
                        ):
                            st.session_state["selected_coffee_file"] = (
                                str(file.resolve())
                            )
                            st.session_state.pop("last_prediction", None)

                    except Exception as error:
                        st.warning(
                            f"Gambar {file.name} tidak dapat dibuka: "
                            f"{error}"
                        )

        selected_path = st.session_state.get(
            "selected_coffee_file"
        )

        if selected_path:
            selected_file = Path(selected_path)

            try:
                # Pastikan gambar terpilih berasal dari folder test.
                selected_file.resolve().relative_to(
                    DATASET_DIR.resolve()
                )

                if selected_file.is_file():
                    image = Image.open(selected_file).convert("RGB")
                    st.success(
                        f"Foto terpilih: {selected_file.parent.name} — "
                        f"{selected_file.name}"
                    )
                    st.image(
                        image,
                        caption="Foto yang akan dianalisis",
                        width=200,
                    )
                else:
                    st.warning("File foto tidak ditemukan.")

            except (ValueError, OSError):
                st.warning("Lokasi foto tidak valid.")
                st.session_state.pop("selected_coffee_file", None)

else:
    uploaded_file = st.file_uploader(
        "Pilih gambar dari komputer",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Foto yang akan dianalisis",
                width=200,
            )

            # Hapus pilihan dataset saat pengguna beralih ke upload.
            st.session_state.pop("selected_coffee_file", None)

        except Exception as error:
            st.error(f"Gambar tidak dapat dibuka: {error}")


# =====================================
# 6. PENGOLAHAN CITRA: CONTRAST
# =====================================
if image is not None:
    st.divider()
    st.subheader("2. Pengolahan Citra: Contrast")

    contrast = st.slider(
        "Tingkat kontras",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help=(
            "Nilai 1.0 mempertahankan kontras asli. "
            "Nilai di bawah 1 mengurangi kontras, "
            "sedangkan nilai di atas 1 meningkatkan kontras."
        ),
    )

    enhancer = ImageEnhance.Contrast(image)
    processed_image = enhancer.enhance(contrast)

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Gambar Asli**")
        st.image(image, width=250)

    with col2:
        st.write("**Hasil Pengolahan**")
        st.image(processed_image, width=250)

    # =====================================
    # 7. PREDIKSI AI
    # =====================================
    st.divider()
    st.subheader("3. Prediksi Tingkat Sangrai dengan AI")

    if st.button(
        "🔍 Analisis Tingkat Sangrai",
        type="primary",
    ):
        with st.spinner("Model sedang menganalisis gambar..."):
            try:
                ai_image = image.resize((224, 224))

                image_array = np.asarray(
                    ai_image,
                    dtype=np.float32,
                )
                image_array = np.expand_dims(
                    image_array,
                    axis=0,
                )

                # Input harus sesuai dengan format saat model dilatih.
                predictions = model.predict(
                    image_array,
                    verbose=0,
                )[0]

                predicted_index = int(np.argmax(predictions))

                if isinstance(class_names, dict):
                    ordered_names = [
                        str(
                            class_names.get(
                                str(i),
                                class_names.get(i, str(i)),
                            )
                        )
                        for i in range(len(predictions))
                    ]
                else:
                    ordered_names = [
                        str(name) for name in class_names
                    ]

                predicted_class = ordered_names[predicted_index]
                confidence = float(predictions[predicted_index])

                st.session_state["last_prediction"] = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "predictions": predictions.tolist(),
                    "ordered_names": ordered_names,
                }

            except Exception as error:
                st.error(f"Prediksi gagal: {error}")

    # Tampilkan hasil yang tersimpan meskipun slider berubah.
    result = st.session_state.get("last_prediction")

    if result:
        predicted_class = result["predicted_class"]

        st.success(f"Kategori prediksi: {predicted_class}")

        # Cocokkan nama kelas meskipun label model menggunakan
        # huruf kecil, misalnya 'dark'.
        normalized_class = predicted_class.strip().capitalize()
        info = ROAST_INFO.get(normalized_class)

        if info:
            st.markdown(f"### {info['judul']}")
            st.write(info["deskripsi"])

        st.metric(
            "Skor keyakinan model",
            f"{result['confidence'] * 100:.2f}%",
        )

        st.caption(
            "Skor keyakinan adalah keluaran model, "
            "bukan jaminan bahwa prediksi benar."
        )

        st.markdown("#### Skor setiap kategori")

        for index, name in enumerate(result["ordered_names"]):
            score = float(result["predictions"][index])
            st.write(f"{name}: {score * 100:.2f}%")
            st.progress(float(np.clip(score, 0.0, 1.0)))

    # =====================================
    # 8. DOWNLOAD HASIL KONTRAS
    # =====================================
    st.divider()
    st.subheader("4. Unduh Hasil Pengolahan")

    buffer = io.BytesIO()
    processed_image.save(buffer, format="PNG")

    st.download_button(
        label="⬇️ Unduh Gambar Hasil",
        data=buffer.getvalue(),
        file_name="hasil_contrast.png",
        mime="image/png",
    )

else:
    st.info(
        "Pilih foto dari dataset atau unggah gambar "
        "untuk memulai pengolahan dan prediksi."
    )


# =====================================
# 9. FOOTER
# =====================================
st.divider()

st.caption(
    "Coffee Bean Roast Analyzer | "
    "Pengolahan Citra Digital + Klasifikasi AI"
)
