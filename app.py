import streamlit as st
import ee
import geemap
import datetime
import json

# Configuración de la página
st.set_page_config(layout="wide", page_title="CONAF - Monitor de Quemas")

# --- FUNCIONES DE PROCESAMIENTO ---
def maskS2sr(image):
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
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
if "EARTHENGINE_TOKEN" in st.secrets:
    try:
        ee_key_dict = json.loads(st.secrets["EARTHENGINE_TOKEN"])
        credentials = ee.ServiceAccountCredentials(
            ee_key_dict['client_email'], 
            key_data=ee_key_dict['private_key']
        )
        
        # Inicialización forzada
        ee.Initialize(credentials)
        
        # Parche crítico: Inyectamos las credenciales donde geemap las busca
        ee.data._credentials = credentials 
        
    except Exception as e:
        st.error(f"Error al conectar con Earth Engine: {e}")
else:
    st.error("No se encontró el token de Earth Engine en Secrets.")
# --- INTERFAZ DE STREAMLIT ---
st.title("🔥 Sistema de Monitoreo de Quemas Agrícolas")
st.sidebar.header("Parámetros de Análisis")

semana_select = st.sidebar.text_input("Semana ISO (ej: 2025-W14)", "2025-W14")
ind_RdNBR = st.sidebar.slider("Umbral RdNBR", 100, 500, 300)
ndvi_max_post = st.sidebar.slider("NDVI Máx Post-Quema", 0.0, 0.5, 0.3)
area_minima = st.sidebar.number_input("Área mínima (m2)", value=1000)

if st.sidebar.button("Ejecutar Análisis"):
    with st.spinner("Procesando datos satelitales..."):
        datos_base = ee.FeatureCollection("projects/conaf-478214/assets/Quemas_Final_Limpio_4326")
        
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

            pre_start, pre_end = min_inicio.advance(-14, 'day'), min_inicio.advance(-1, 'day')
            post_start, post_end = max_control.advance(4, 'day'), max_control.advance(21, 'day')

            base_sat = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
                        .filterBounds(boundingBox)\
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))\
                        .map(maskS2sr).map(get_INDEX_S2).map(renameBandsS2)

            pref = base_sat.filterDate(pre_start, pre_end).median().clip(boundingBox)
            postf = base_sat.filterDate(post_start, post_end).median().clip(boundingBox)

            # Cálculo de RdNBR con nombres seguros
            firesev = pref.select(['NBR'],['NBR_pre']).addBands(postf.select(['NBR'],['NBR_post']))
            RdNBR = firesev.expression("(b('NBR_pre') - b('NBR_post')) / (sqrt(abs(b('NBR_pre') / 1000) + 0.001))").toFloat().rename('RdNBR')

            propertyMask = ee.Image(0).byte().paint(inc_semana, 1)
            firemask = RdNBR.gt(ind_RdNBR).updateMask(propertyMask).selfMask()
            
            firevect = firemask.addBands(postf.select('NDVI')).reduceToVectors(
                geometry=boundingBox, scale=10, geometryType='polygon', 
                eightConnected=False, labelProperty='Incendio', reducer=ee.Reducer.mean(), maxPixels=1e13
            )

            firevect_final = firemask.multiply(ee.Image.pixelArea()).reduceRegions(
                collection=firevect, reducer=ee.Reducer.sum(), scale=10
            ).filter(ee.Filter.lt('mean', ndvi_max_post)).filter(ee.Filter.gt('sum', area_minima))

            # --- VISUALIZACIÓN ---
            # ee_initialize=False es CRÍTICO aquí
            m = geemap.Map(ee_initialize=False)
            m.centerObject(boundingBox, 12)
            
            vis_rgb = {'bands': ['R', 'G', 'B'], 'min': 0, 'max': 2500, 'gamma': 1.4}
            m.add_ee_layer(pref, vis_rgb, 'Imagen PRE')
            m.add_ee_layer(postf, vis_rgb, 'Imagen POST')
            m.add_ee_layer(firemask, {'palette': ['red']}, 'Áreas Quemadas (Raster)')
            m.add_ee_layer(firevect_final, {'color': 'cyan'}, 'Quemas por Predio (Vector)')
            
            m.to_streamlit(height=700)
            st.write(f"Resultados encontrados: {firevect_final.size().getInfo()} polígonos.")
