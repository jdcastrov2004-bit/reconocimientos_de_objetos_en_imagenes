import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys
from PIL import Image

st.set_page_config(page_title="Detección de Objetos en Tiempo Real", page_icon="🔍", layout="wide")

@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        Recomendaciones:
        1. Instalar una versión compatible de PyTorch y YOLOv5:
           ```
           pip install torch==1.12.0 torchvision==0.13.0
           pip install yolov5==7.0.9
           ```
        2. Asegúrate de tener el archivo del modelo en la ubicación correcta
        3. Si el problema persiste, intenta descargar el modelo directamente de torch hub
        """)
        return None

st.title("🔍 Detección de Objetos en Imágenes")
image = Image.open("deteccion.jpg")
st.image(image, width=360)
st.write(
    "En esta actividad capturarás una imagen con tu cámara y aplicarás un modelo **YOLOv5** para detectar objetos. "
    "Ajusta los parámetros en la barra lateral y observa cómo cambian los resultados, así como el conteo por categoría."
)

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

if model:
    st.sidebar.title("Parámetros")
    with st.sidebar:
        st.subheader("Configuración de detección")
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")
        st.subheader('Opciones avanzadas')
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False)
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False)
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10)
        except:
            st.warning("Algunas opciones avanzadas no están disponibles con esta configuración")

    main_container = st.container()
    with main_container:
        picture = st.camera_input("Capturar imagen", key="camera")
        if picture:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            with st.spinner("Detectando objetos..."):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()
            try:
                try:
                    predictions = results.pred[0]
                except AttributeError:
                    predictions = results.xyxy[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Imagen con detecciones")
                    try:
                        results.render()
                        annotated = results.imgs[0]
                        st.image(annotated[:, :, ::-1], use_container_width=True)
                    except:
                        st.image(cv2_img, channels='BGR', use_container_width=True)
                with col2:
                    st.subheader("Objetos detectados")
                    try:
                        label_names = results.names
                    except AttributeError:
                        label_names = getattr(model, "names", {})
                    category_count = {}
                    for category in categories:
                        idx = int(category.item()) if hasattr(category, 'item') else int(category)
                        category_count[idx] = category_count.get(idx, 0) + 1
                    data = []
                    for category, count in category_count.items():
                        mask = (categories == category).cpu().numpy() if hasattr(categories, "cpu") else (categories == category)
                        conf_vals = scores[mask] if isinstance(mask, np.ndarray) else scores[categories == category]
                        conf_mean = float(conf_vals.mean().item() if hasattr(conf_vals.mean(), "item") else conf_vals.mean()) if len(conf_vals) > 0 else 0.0
                        label = label_names[category] if isinstance(label_names, dict) and category in label_names else str(category)
                        data.append({"Categoría": label, "Cantidad": count, "Confianza promedio": f"{conf_mean:.2f}"})
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index('Categoría')['Cantidad'])
                    else:
                        st.info("No se detectaron objetos con los parámetros actuales.")
                        st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")
            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")
                st.stop()
else:
    st.error("No se pudo cargar el modelo. Por favor verifica las dependencias e inténtalo nuevamente.")
    st.stop()

st.markdown("---")
st.caption("Esta aplicación utiliza YOLOv5 para detección de objetos. Desarrollada con Streamlit y PyTorch.")
