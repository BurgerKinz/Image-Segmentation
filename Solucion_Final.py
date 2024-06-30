import rasterio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# Función para cargar la imagen satelital
def cargar_imagen(ruta_de_imagen):
    try:
        with rasterio.open(ruta_de_imagen) as imagen:
            return imagen.read(), imagen.profile
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return None, None

# Función para preprocesar los datos
def preprocesar_datos(imagen_satelital):
    scaler = StandardScaler()
    imagen_satelital_plana = imagen_satelital.reshape((imagen_satelital.shape[0], -1)).T
    return scaler.fit_transform(imagen_satelital_plana)

# Función para aplicar KMeans y evaluar los resultados
def aplicar_kmeans_evaluar(imagen_escalada, metadata, num_clusters, colormap, imagen_original, ruta_de_imagen):
    print(f"Aplicando KMeans con {num_clusters} clusters y colormap {colormap}")
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
    imagen_kmeans = kmeans.fit_predict(imagen_escalada)
    imagen_kmeans = imagen_kmeans.reshape((metadata['height'], metadata['width']))
    
    # Crear carpeta para guardar las imágenes
    nombre_carpeta = f"{os.path.splitext(ruta_de_imagen)[0]}_resultados"
    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)

    # Guardar la imagen original
    plt.figure(figsize=(6, 6))
    plt.imshow(np.moveaxis(imagen_original, 0, -1))
    plt.title('Imagen Original')
    plt.axis('off')
    plt.savefig(os.path.join(nombre_carpeta, 'imagen_original.png'))
    plt.show()

    # Mostrar la segmentación con KMeans
    plt.figure(figsize=(12, 6))
    plt.subplot(1, num_clusters + 1, 1)
    plt.imshow(imagen_kmeans, cmap=colormap)
    plt.title(f'Imagen Segmentada con {num_clusters} clusters')
    plt.axis('off')
    
    # Mostrar y guardar las capas por separado
    for i in range(num_clusters):
        plt.subplot(1, num_clusters + 1, i + 2)
        plt.imshow(imagen_kmeans == i, cmap='cividis')
        plt.title(f'Capa {i + 1}')
        plt.axis('off')
        
        # Guardar cada capa como imagen separada
        nombre_archivo_capa = f"{nombre_carpeta}/capa_{i + 1}.png"
        plt.imsave(nombre_archivo_capa, imagen_kmeans == i, cmap='cividis')

    # Guardar la figura completa
    nombre_archivo_segmentada = f"{nombre_carpeta}/imagen_segmentada_{num_clusters}_{colormap}.png"
    plt.savefig(nombre_archivo_segmentada)
    plt.show()

    return imagen_kmeans  # Devolvemos la imagen segmentada

# Función para aplicar el método del codo y determinar los números óptimos de clusters
def metodo_del_codo(imagen_escalada, max_clusters):
    inercia = []
    for k in range(1, max_clusters + 1):
        print(f"Calculando KMeans para {k} clusters (método del codo)")
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        kmeans.fit(imagen_escalada)
        inercia.append(kmeans.inertia_)
    
    # Determinar los números óptimos de clusters
    dif_inercia = np.diff(inercia)
    dif_dif_inercia = np.diff(dif_inercia)
    num_clusters_optimos = np.argsort(dif_dif_inercia)[-2:] + 2  # Tomamos los dos últimos índices
    return num_clusters_optimos

# Establecer la semilla aleatoria para el colormap
np.random.seed(7)

# Ruta de la imagen satelital
ruta_de_imagen = "Valdivia.jpg"

# Cargar la imagen y obtener metadatos
imagen_satelital, metadata = cargar_imagen(ruta_de_imagen)
if imagen_satelital is not None:

    # Preprocesar los datos
    imagen_escalada = preprocesar_datos(imagen_satelital)

    # Aplicar el método del codo para determinar números óptimos de clusters
    print("Aplicando el método del codo")
    num_clusters_codo = metodo_del_codo(imagen_escalada, max_clusters=5)  # Reducción de max_clusters para prueba inicial
    print(f"Números óptimos de clusters según el método del codo: {num_clusters_codo}")

    # Preguntar al usuario cuál número de clusters quiere utilizar
    try:
        num_clusters_usuario = int(input(f"Ingrese el número de clusters que desea utilizar (opciones: {num_clusters_codo}): "))
        if num_clusters_usuario in num_clusters_codo:
            colormap_usuario = input("Ingrese el colormap que desea utilizar (opciones: 'viridis', 'inferno'): ").strip().lower()
            if colormap_usuario not in ['viridis', 'inferno']:
                print("Colormap ingresado no válido. Se utilizará 'viridis' por defecto.")
                colormap_usuario = 'viridis'
        else:
            print("El número de clusters ingresado no está en las opciones válidas.")
            colormap_usuario = 'viridis'  # Colormap por defecto si no se elige uno válido
        
        # Aplicar KMeans con los parámetros elegidos por el usuario
        imagen_segmentada = aplicar_kmeans_evaluar(imagen_escalada, metadata, num_clusters_usuario, colormap_usuario, imagen_satelital, ruta_de_imagen)
    
    except ValueError:
        print("Entrada inválida. Se utilizarán los números óptimos de clusters determinados por el método del codo y 'viridis' como colormap.")
else:
    print("No se pudo cargar la imagen.")
