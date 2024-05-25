import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans

# Función para segmentar una imagen utilizando K-Means y determinar el número óptimo de clusters
def segmentar_imagen_auto(imagen):
    # Redimensionar la imagen a una matriz 2D
    nrows, ncols, nchannels = imagen.shape
    imagen_2d = imagen.reshape(nrows * ncols, nchannels)
    
    # Inicializar una lista para almacenar las inercias
    inercias = []
    
    # Probar diferentes números de clusters
    for num_clusters in range(2, 11):  # Prueba desde 2 hasta 10 clusters
        # Aplicar K-Means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(imagen_2d)
        
        # Calcular la inercia y agregarla a la lista
        inercias.append(kmeans.inertia_)
    
    # Elbow Method: Elegir el número óptimo de clusters
    # Buscar el punto de inflexión en la curva del codo
    deltas = np.diff(np.diff(inercias))
    optimal_num_clusters = np.argmax(deltas) + 2  # El índice de la máxima diferencia
    
    # Segmentar la imagen con el número óptimo de clusters
    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
    kmeans.fit(imagen_2d)
    segmented_image_1d = kmeans.labels_
    segmented_image = segmented_image_1d.reshape(nrows, ncols)
    
    return segmented_image

# Cargar la imagen
imagen_path = "C:\\Users\\cduty\\Analisis de Algoritmos\\Santiago_Dominicos.png"
imagen_satelital = io.imread(imagen_path)

# Segmentar la imagen automáticamente
imagen_segmentada_auto = segmentar_imagen_auto(imagen_satelital)

# Lista de colormaps a probar
colormaps = ['viridis', 'inferno']

# Mostrar la imagen original y las imágenes segmentadas con diferentes colormaps
plt.figure(figsize=(15, 10))

# Mostrar la imagen original
plt.subplot(2, len(colormaps) + 1, len(colormaps) + 1)
plt.imshow(imagen_satelital)
plt.axis('off')
plt.title('Imagen Original')

# Generar y mostrar la imagen segmentada con cada colormap
for i, cmap in enumerate(colormaps, start=1):
    plt.subplot(2, len(colormaps) + 1, i)
    plt.imshow(imagen_segmentada_auto, cmap=cmap)
    plt.axis('off')
    plt.title(f'Segmentada - {cmap}')

plt.tight_layout()
plt.show()
