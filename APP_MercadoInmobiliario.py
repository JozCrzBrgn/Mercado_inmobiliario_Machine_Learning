import pandas as pd
import numpy as np
import pickle
import streamlit as st
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent='Pruebas')

model = pickle.load(open('Proyecto_5\model_RFC.sav', 'rb'))

##############
# Page Title #
##############
st.title('Clasificación del Mercado Inmobiliario')

st.markdown("""
Se realizó un modelo de machine learning para realizar una predicción sobre si una casa es barata o cara.

* **Librerías Python:** streamlit, pandas, geopy.
* **Proyecto en GitHub:** [Jupyter Notebook](https://github.com/JozCrzBrgn/Mercado_inmobiliario_Machine_Learning/blob/main/Mercado_Inmobiliario.ipynb).
""")

###############
# Page Inputs #
###############
st.sidebar.header('Parámetros de entrada')
provincia = st.sidebar.selectbox('Provincia', ['Antioquia', 'Atlántico', 'Cundinamarca', 'Valle del Cauca', 'Santander'])
if provincia=='Antioquia':
    ciudad = st.sidebar.selectbox('Ciudad', ['Medellín'])
    barrio = st.sidebar.selectbox('Barrio', ['San Antonio de Prado','El Poblado','Candelaria','San Javier',
                                             'Laureles','Aranjuez','Belén','Santa Elena','Altavista','Popular',
                                             'La América','San Cristóbal','Villa Hermosa','Buenos Aires','Guayabal',
                                             'Castilla','Robledo','Doce de Octubre','Manrique','Santa Cruz','Palmitas'])
elif provincia=='Atlántico':
    ciudad = st.sidebar.selectbox('Ciudad', ['Barranquilla'])
    barrio = st.sidebar.selectbox('Barrio',['San Felipe','Campo Alegre','Nuevo Horizonte','Norte-Centro Histórico',
                                            'Puerto Colombia','Carrizal','Nueva Granada','Soledad','Ríomar',
                                            'Paseo de la Castellana','Olaya','El Recreo','Las Palmas'])
elif provincia=='Cundinamarca': # Corregir zonas
    ciudad = st.sidebar.selectbox('Ciudad', ['Soacha', 'Bogotá D.C.'])
    if ciudad=='Soacha':
        barrio = st.sidebar.selectbox('Barrio', ['San Mateo'])
    else:
        barrio = st.sidebar.selectbox('Barrio', ['Zona Noroccidental','Zona Norte','Zona Chapinero',
                                                 'Zona Suroccidental','Zona Sur','Zona Occidental','Zona Centro'])
elif provincia=='Valle del Cauca':
    ciudad = st.sidebar.selectbox('Ciudad', ['Cali'])
    barrio = st.sidebar.selectbox('Barrio', ['San Fernando Viejo','El Ingenio','Lili','Pance','La Flora',
                                             'Ciudad Jardín','Santa Mónica','San Fernando Nuevo','Caney',
                                             'Santa Isabel','El Limonar'])
elif provincia=='Santander':
    ciudad = st.sidebar.selectbox('Ciudad', ['Bucaramanga'])
    barrio = st.sidebar.selectbox('Barrio', ['San Alonso','El Prado','Mejoras Públicas','Alarcón','Antonia Santos'])

tipo_prop = st.sidebar.selectbox('Tipo de propiedad', ['Casa', 'Apartamento', 'Otro', 'Oficina', 'Finca', 'Lote',
                                                       'Local comercial', 'Parqueadero'])

cuartos = st.sidebar.slider('Número de Cuartos', 1, 20, 20)
banios = st.sidebar.slider('Número de Baños', 1, 20, 20)
ambientes = st.sidebar.slider('Número de Ambientes', 1, 20, 20)

################
# Page Outputs #
################
def Inputs(provincia, ciudad, barrio, tipo_prop, cuartos, banios, ambientes):
    col_names = ['Antioquia', 'Arauca', 'Atlántico', 'Bolívar', 'Boyacá', 'Caldas', 'Caquetá', 'Casanare', 'Cauca', 'Cesar', 'Chocó',
                 'Cundinamarca', 'Córdoba', 'Guainía', 'Guaviare', 'Huila', 'La Guajira', 'Magdalena', 'Meta', 'Nariño',
                 'Norte de Santander', 'Putumayo', 'Quindío', 'Risaralda', 'San Andrés Providencia y Santa Catalina', 'Santander', 'Sucre',
                 'Tolima', 'Valle del Cauca', 'Vichada', 'Alarcón', 'Altavista', 'Antonia Santos', 'Aranjuez', 'Belén', 'Buenos Aires',
                 'Campo Alegre', 'Candelaria', 'Caney', 'Carrizal', 'Castilla', 'Ciudad Jardín', 'Doce de Octubre', 'El Ingenio', 'El Limonar',
                 'El Poblado', 'El Prado', 'El Recreo', 'Guayabal', 'La América', 'La Flora', 'Las Palmas', 'Laureles', 'Lili', 'Manrique',
                 'Mejoras Públicas', 'Norte-Centro Histórico', 'Nueva Granada', 'Nuevo Horizonte', 'Olaya', 'Palmitas', 'Pance',
                 'Paseo de la Castellana', 'Popular', 'Puerto Colombia', 'Robledo', 'Ríomar', 'San Alonso', 'San Antonio de Prado',
                 'San Cristóbal', 'San Felipe', 'San Fernando Nuevo', 'San Fernando Viejo', 'San Javier', 'San Mateo', 'Santa Cruz',
                 'Santa Elena', 'Santa Isabel', 'Santa Mónica', 'Soledad', 'Villa Hermosa', 'Zona Centro', 'Zona Chapinero',
                 'Zona Noroccidental', 'Zona Norte', 'Zona Occidental', 'Zona Sur', 'Zona Suroccidental', 'Abejorral', 'Acacías', 'Acandí',
                 'Agua de Dios', 'Aguazul', 'Aipe', 'Albania', 'Albán', 'Alvarado', 'Anapoima', 'Andalucía', 'Anolaima', 'Anserma', 'Apulo',
                 'Arauca', 'Arbeláez', 'Arcabuco', 'Arjona', 'Armenia', 'Balboa', 'Baranoa', 'Barbosa', 'Barichara', 'Barrancabermeja',
                 'Barranquilla', 'Bello', 'Bochalema', 'Bogotá D.C.', 'Bojacá', 'Bucaramanga', 'Buenaventura', 'Buesaco', 'Cachipay',
                 'Caicedonia', 'Cajicá', 'Calarca', 'Caldas', 'Cali', 'Calima', 'Candelaria', 'Caparrapí', 'Carmen de Apicalá',
                 'Carmen de Carupa', 'Cartagena', 'Cartago', 'Casabianca', 'Chaparral', 'Chigorodó', 'Chinauta', 'Chinácota', 'Chipaque',
                 'Chiquinquirá', 'Choachí', 'Chocontá', 'Chía', 'Cimitarra', 'Circasia', 'Clemencia', 'Coello', 'Cogua', 'Colombia',
                 'Copacabana', 'Cota', 'Coveñas', 'Cucunubá', 'Cunday', 'Curití', 'Cúcuta', 'Dagua', 'Dibulla', 'Duitama', 'Durania',
                 'El Carmen de Viboral', 'El Cerrito', 'El Colegio', 'El Peñón', 'El Rosal', 'Envigado', 'Espinal', 'Facatativá', 'Filandia',
                 'Flandes', 'Florencia', 'Floridablanca', 'Fredonia', 'Fresno', 'Funza', 'Fusagasugá', 'Gachancipá', 'Galapa', 'Garzón',
                 'Girardot', 'Girardota', 'Girón', 'Granada', 'Guacarí', 'Guadalajara de Buga', 'Guaduas', 'Guamal', 'Guamo', 'Guarne',
                 'Guasca', 'Guatapé', 'Guatavita', 'Guateque', 'Guayabal de Siquima', 'Gómez Plata', 'Hispania', 'Honda', 'Ibagué', 'Icononzo',
                 'Inírida', 'Ipiales', 'Itagui', 'Jamundí', 'Jardín', 'Jenesano', 'Juan de Acosta', 'La Calera', 'La Ceja', 'La Cumbre',
                 'La Dorada', 'La Estrella', 'La Mesa', 'La Plata', 'La Tebaida', 'La Vega', 'La Virginia', 'Lebríja', 'Los Patios',
                 'Los Santos', 'Lérida', 'Líbano', 'Macheta', 'Madrid', 'Malambo', 'Manizales', 'Manzanares', 'Maní', 'Marinilla', 'Mariquita',
                 'Marsella', 'Medellín', 'Medina', 'Melgar', 'Moniquirá', 'Montenegro', 'Montería', 'Mosquera', 'Neira', 'Neiva', 'Nemocón',
                 'Nilo', 'Nocaima', 'Orocué', 'Pacho', 'Paipa', 'Palermo', 'Palestina', 'Palmar de Varela', 'Palmira', 'Pandi', 'Pasto',
                 'Paz de Ariporo', 'Pereira', 'Piedecuesta', 'Piedras', 'Pinchote', 'Pitalito', 'Planeta Rica', 'Popayán', 'Pore', 'Pradera',
                 'Prado', 'Providencia', 'Puerto Boyacá', 'Puerto Carreño', 'Puerto Colombia', 'Puerto Gaitán', 'Puerto Lleras', 'Puerto López',
                 'Puerto Parra', 'Puerto Salgar', 'Puerto Triunfo', 'Puerto Wilches', 'Purificación', 'Quebradanegra', 'Quimbaya', 'Restrepo',
                 'Retiro', 'Ricaurte', 'Riofrío', 'Riohacha', 'Rionegro', 'Rivera', 'Sabanagrande', 'Sabanalarga', 'Sabaneta', 'Sahagún',
                 'Salamina', 'Saldaña', 'Salento', 'San Alberto', 'San Andrés', 'San Antonio del Tequendama', 'San Carlos de Guaroa',
                 'San Francisco', 'San Gil', 'San Jerónimo', 'San Juan de Arama', 'San Juan de Rioseco', 'San Luis', 'San Marcos',
                 'San Martín', 'San Onofre', 'San Pedro de Uraba', 'San Rafael', 'Santa Catalina', 'Santa Marta', 'Santa Rosa',
                 'Santa Rosa de Cabal', 'Santa Sofía', 'Santafé de Antioquia', 'Santiago de Tolú', 'Sasaima', 'Sesquilé', 'Sibaté',
                 'Silvania', 'Simijaca', 'Sincelejo', 'Soacha', 'Socorro', 'Sogamoso', 'Soledad', 'Sopetrán', 'Sopó', 'Subachoque', 'Suesca',
                 'Supatá', 'Sutamarchán', 'Sutatausa', 'Suárez', 'Tabio', 'Tauramena', 'Tausa', 'Tena', 'Tenjo', 'Tibacuy', 'Tibasosa',
                 'Tocaima', 'Tocancipá', 'Trujillo', 'Tubará', 'Tuluá', 'Tunja', 'Turbaco', 'Turbaná', 'Turbo', 'Valledupar', 'Venadillo',
                 'Venecia', 'Vergara', 'Vianí', 'Victoria', 'Vijes', 'Villa de Leyva', 'Villa del Rosario', 'Villamaría', 'Villanueva',
                 'Villapinzón', 'Villavicencio', 'Villeta', 'Viotá', 'Viterbo', 'Yacopí', 'Yopal', 'Yotoco', 'Yumbo', 'Zapatoca', 'Zarzal',
                 'Zipacón', 'Zipaquirá', 'Apartamento', 'Casa', 'Finca', 'Local comercial', 'Lote', 'Oficina', 'Otro', 'Parqueadero']

    def maxmin(cuarto, banio, amb):
        max_cuarto = 2.764391e+07
        max_banio = 125.000000
        max_amb = 2.764392e+07
        return [(cuarto/max_cuarto), (banio/max_banio), (amb/max_amb)]

    # Lista de valores nulos
    cat = [0 for i in range(len(col_names))]
    list_pcb = [provincia, barrio, ciudad, tipo_prop]
    # Ubicacion de valores
    for L in list_pcb:
        idx = col_names.index(L)
        cat[idx] = 1.0
    # Escalamiento de variables numéricas
    num = maxmin(cuartos, banios, ambientes)
    # Transformación a array
    X = np.array([cat + num])
    return X

def Clasif(y_pred):
    # barata (0) y cara (1)
    if y_pred[0]==1:
        return 'Cara'
    else:
        return 'Barata'

if st.button('Clasificar'):
    st.subheader('Ubicación de la zona.')
    location = geolocator.geocode(str(provincia) + ", " + str(ciudad) + ", " + str(barrio) + ", Colombia")
    df = pd.DataFrame([[location.latitude, location.longitude]], columns=['lat', 'lon'])
    st.map(df, zoom=12)

    st.subheader('Clasificación del inmueble.')

    X_pred = Inputs(provincia, ciudad, barrio, tipo_prop, cuartos, banios, ambientes)
    y_pred = model.predict(X_pred)
    clasif = Clasif(y_pred)

    data = {'Provincia': [provincia],
            'Ciudad': [ciudad],
            'Barrio':[barrio],
            'Tipo de Propiedad':[tipo_prop],
            'Cuarto/s':[cuartos],
            'Baño/s':[banios],
            'Ambiente/s':[ambientes],
            'Clasificación':[clasif]}
    df = pd.DataFrame.from_dict(data)
    st.write(df)


# streamlit run "Proyecto_5\APP_MercadoInmobiliario.py"