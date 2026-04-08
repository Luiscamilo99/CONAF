import streamlit as st
import ee
import geemap
import datetime

# Configuración de la página
st.set_page_config(layout="wide", page_title="CONAF - Monitor de Quemas")

# --- FUNCIONES DE PROCESAMIENTO ---
def maskS2sr(image):
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).and_(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).copyProperties(image, ["system:time_start"])

def get_INDEX_S2(image):
    ndvi = image.expression('(NIR - Red) / (NIR + Red)', {
        'NIR': image.select('B8'),
        'Red': image.select('B4')}).rename('NDVI')
    nbr = image.expression('((NIR - SWIR2) / (NIR + SWIR2))*1000', {
        'NIR': image.select('B8'),
        'SWIR2': image.select('B12')}).rename('NBR')
    return image.addBands([ndvi, nbr]).copyProperties(image, ['system:time_start', 'system:time_end'])

def renameBandsS2(image):
    bands = ['B2', 'B3', 'B4', 'B8','B11','B12', 'NDVI', 'NBR']
    new_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NBR']
    return image.select(bands).rename(new_bands)

# --- INICIALIZACIÓN DE EE ---
import json

if "EARTHENGINE_TOKEN" in st.secrets:
    try:
        # Cargamos el JSON desde los Secrets de Streamlit
        ee_key_dict = json.loads(st.secrets["EARTHENGINE_TOKEN"])
        
        # Creamos las credenciales
        credentials = ee.ServiceAccountCredentials(
            ee_key_dict['client_email'], 
            key_data=ee_key_dict['private_key']
        )
        
        # Inicializamos con las credenciales
        ee.Initialize(credentials)
    except Exception as e:
        st.error(f"Error al conectar con Earth Engine: {e}")
else:
    st.error("No se encontró el token de Earth Engine. Configura los 'Secrets' en Streamlit.")
# --- INTERFAZ DE STREAMLIT ---
st.title("🔥 Sistema de Monitoreo de Quemas Agrícolas")
st.sidebar.header("Parámetros de Análisis")

# Selección de Semana (Simplificado para la App)
semana_select = st.sidebar.text_input("Semana ISO (ej: 2025-W14)", "2025-W14")
ind_RdNBR = st.sidebar.slider("Umbral RdNBR", 100, 500, 300)
ndvi_max_post = st.sidebar.slider("NDVI Máx Post-Quema", 0.0, 0.5, 0.3)
area_minima = st.sidebar.number_input("Área mínima (m2)", value=1000)

if st.sidebar.button("Ejecutar Análisis"):
    with st.spinner("Procesando datos satelitales..."):
        # 1. Cargar Datos Base
        datos_base = ee.FeatureCollection("projects/conaf-478214/assets/Quemas_Final_Limpio_4326")
        
        # Función para preparar fechas (Traducida a Python)
        def prepararFechas(feat):
            fechaInicio = ee.Date.fromYMD(feat.getNumber('inicion'), feat.getNumber('inicims'), feat.getNumber('iniciod'))
            fechaControl = ee.Date.fromYMD(feat.getNumber('contrln'), feat.getNumber('cntrlms'), feat.getNumber('contrld'))
            año = fechaInicio.get('year')
            inicioSemana = fechaInicio.advance(fechaInicio.getRelative('day', 'week').subtract(1).mod(7).multiply(-1), 'day')
            periodo = año.format('%d').cat('-W').cat(inicioSemana.getRelative('week', 'year').add(1).format('%02d'))
            return feat.set({'fecha_inicio_ms': fechaInicio.millis(), 'fecha_control_ms': fechaControl.millis(), 'periodo_semanal': periodo})

        datosLimpios = datos_base.filter(ee.Filter.notNull(['inicion', 'contrln'])).map(prepararFechas)
        inc_semana = datosLimpios.filter(ee.Filter.eq('periodo_semanal', semana_select))
        
        if inc_semana.size().getInfo() == 0:
            st.warning(f"No hay registros para la semana {semana_select}")
        else:
            boundingBox = inc_semana.geometry().bounds()
            min_inicio = ee.Date(inc_semana.aggregate_min('fecha_inicio_ms'))
            max_control = ee.Date(inc_semana.aggregate_max('fecha_control_ms'))

            # Ventanas temporales
            pre_start, pre_end = min_inicio.advance(-14, 'day'), min_inicio.advance(-1, 'day')
            post_start, post_end = max_control.advance(4, 'day'), max_control.advance(21, 'day')

            # Procesamiento Satelital
            base_sat = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                        .filterBounds(boundingBox)\
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
                        .map(maskS2sr).map(get_INDEX_S2).map(renameBandsS2)

            pref = base_sat.filterDate(pre_start, pre_end).median().clip(boundingBox)
            postf = base_sat.filterDate(post_start, post_end).median().clip(boundingBox)

            # RdNBR
            firesev = pref.select('NBR').addBands(postf.select('NBR'))
            RdNBR = firesev.expression("(b('NBR') - b('NBR_1')) / (sqrt(abs(b('NBR') / 1000) + 0.001))").toFloat().rename('RdNBR')

            # Máscara de fuego
            propertyMask = ee.Image().byte().paint(inc_semana, 1)
            firemask = RdNBR.gt(ind_RdNBR).updateMask(propertyMask).selfMask()
            
            # Vectorización
            firevect = firemask.addBands(postf.select('NDVI')).reduceToVectors(
                geometry=boundingBox, scale=10, geometryType='polygon', 
                eightConnected=False, labelProperty='Incendio', reducer=ee.Reducer.mean(), maxPixels=1e13
            )

            # Filtrado por área y NDVI
            firevect_final = firemask.multiply(ee.Image.pixelArea()).reduceRegions(
                collection=firevect, reducer=ee.Reducer.sum(), scale=10
            ).filter(ee.Filter.lt('mean', ndvi_max_post)).filter(ee.Filter.gt('sum', area_minima))

            # --- VISUALIZACIÓN ---
            m = geemap.Map()
            m.centerObject(boundingBox, 12)
            
            vis_rgb = {'bands': ['R', 'G', 'B'], 'min': 0, 'max': 2500, 'gamma': 1.4}
            m.addLayer(pref, vis_rgb, 'Imagen PRE')
            m.addLayer(postf, vis_rgb, 'Imagen POST')
            m.addLayer(firemask, {'palette': ['red']}, 'Áreas Quemadas (Raster)')
            m.addLayer(firevect_final, {'color': 'cyan'}, 'Quemas por Predio (Vector)')
            
            m.to_streamlit(height=700)
            
            # Botón para descargar como GeoJSON (Streamlit friendly)
            st.write(f"Resultados encontrados: {firevect_final.size().getInfo()} polígonos.")
