import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pickle
import seaborn as sns
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering


st.header("Gender Pay Gap Analysis")

def load_data():
    data = pd.read_csv("data cleaned.csv")
    return data

# Business Understanding
def business_understanding():
    st.title("Business Understanding")

    
    st.markdown(
        """
        <div style="text-align: justify;">
        <img src="https://images.squarespace-cdn.com/content/v1/575c7d10044262e4c49720f7/cf4ec0a7-1c53-4179-9d2a-2508bd157411/107208316-1678801916268-gettyimages-1371442941-gender-pay-gap-balance.jpeg" style="width:100%;height:auto;" alt="Business Understanding Image">

     <h2>Business Objective</h2>
        <p>Tujuan utama dari analisis ini adalah untuk mengindentifikasi kesenjangan gaji berdasarkan jenis kelamin di berbagai negara dan industri di Eropa dari tahun 2010 hingga 2021. Dengan memahami pola-pola yang ada, analisis ini dapat memberikan insight yang lebih mendalam mengenai faktor-faktor yang mendasarinya. Tujuannya adalah untuk mengidentifikasi kelompok negara dengan karakteristik serupa dalam hal kesenjangan gaji gender, serta faktor-faktor sosial-ekonomi yang mempengaruhi, seperti PDB per kapita, tingkat urbanisasi, dan struktur industri.</p>

        <h2>Assess Situation</h2>
        <p>Perlu ditingkatkan awareness terkait isu kesenjangan gaji berdasarkan jenis kelamin di Eropa, sehingga dibutuhkan pemahaman yang lebih dalam tentang untuk menganalisis pola dan karakteristiknya dengan lebih rinci.  Dengan menggunakan teknik pengelompokan, kita dapat mengidentifikasi kelompok negara dengan pola kesenjangan gaji gender yang serupa.</p>

        <h2>Data Mining Goals</h2>
        <p>Tujuan data mining adalah untuk mengembangkan model clustering untuk mengelompokkan negara-negara Eropa berdasarkan pola kesenjangan gaji gender, menyelidiki apakah ada pola kesenjangan gaji gender yang lebih menonjol dalam sektor industri tertentu dan kelompok negara tertentu, memberikan rekomendasi kebijakan atau strategi organisasi yang sesuai berdasarkan temuan dari model clustering.</p>

        <h2>Project Plan</h2>
        <p>Rencana proyek untuk menganalisis dataset ini diawali dengan mengumpulkan dataset yang.Setelah dataset terkumpul, langkah selanjutnya adalah melakukan pembersihan dan pra-pemrosesan data yang diperlukan untuk memastikan keakuratan dan konsistensi dalam analisis selanjutnya. Setelah data dipersiapkan, analisis data eksploratif dilakukan untuk memahami distribusi dan tren data, serta hubungan antara variabel yang relevan. Kemudian, memasuki tahap pengembangan model clustering dengan memilih algoritma yang sesuai untuk mengelompokkan negara-negara berdasarkan pola kesenjangan gaji gender. Setelah model clustering dikembangkan, tahap evaluasi dimulai. Yang terakhir, deployment yang berisi hasil interpretasi pengelompokan untuk mengidentifikasi kelompok negara dengan karakteristik serupa dalam hal kesenjangan gaji gender. Selain itu, kami menganalisis faktor-faktor yang mungkin mempengaruhi pola kesenjangan gaji gender dalam setiap kelompok.</p>
        </div>
        """,
        unsafe_allow_html=True
    )



# Visualisasi Data
def visualize_data(data):
    st.title("Visualisasi Data")
    st.write("Berikut adalah frekuensi data pada setiap kolom:")

    num_cols = [col for col in data.columns if data[col].dtype in ['int64','float64']]

    num_cols_len = len(num_cols)
    num_cols_per_row = 5

    num_rows = int(num_cols_len / num_cols_per_row) + (num_cols_len % num_cols_per_row > 0)
    fig, axs = plt.subplots(num_rows, num_cols_per_row, figsize=(20, 4*num_rows), gridspec_kw={'width_ratios': [1, 1, 1, 1,1]})

    for idx, col in enumerate(num_cols):
        row_idx = int(idx / num_cols_per_row)
        col_idx = idx % num_cols_per_row
        if idx >= num_cols_len:
            axs[row_idx, col_idx].axis('off')
        else:
            axs[row_idx, col_idx].hist(data[col], color="#009999")
            axs[row_idx, col_idx].set_xlabel(col)
            axs[row_idx, col_idx].set_ylabel('Frequency')

    # Remove any empty subplots
    if num_cols_len % num_cols_per_row > 0:
        for i in range(num_cols_len % num_cols_per_row, num_cols_per_row):
            fig.delaxes(axs[num_rows-1, i])

    plt.tight_layout()
    st.pyplot(fig)
    st.write('Histogram di atas adalah menunjukkan pola distribusi data setiap kolom numerik. Puncak histogram menunjukkan di mana sebagian besar nilai-nilai berada. Lebar histogram menunjukkan rentang nilai-nilai yang mungkin.')



import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data_changes(data, selected_country):
    st.title(f"Data untuk Negara {selected_country} dari 2010-2021")
    st.write("""
    **Interpretasi:**
    Visualisasi ini memperlihatkan perubahan nilai numerik di setiap kolom selama periode 2010-2021 untuk negara yang dipilih. Setiap garis pada grafik mewakili tren perubahan nilai dari satu kolom tertentu dari tahun ke tahun.
    
    **Insight:**
    1. **GDP (PDB)**: Jika terdapat peningkatan yang signifikan dalam grafik GDP dari tahun ke tahun, ini menunjukkan pertumbuhan ekonomi yang kuat di negara tersebut. Sebaliknya, penurunan atau stagnasi bisa menjadi sinyal adanya masalah ekonomi yang perlu ditangani.
    
    2. **Urban Population (Populasi Urban)**: Jika grafik menunjukkan peningkatan yang stabil dalam populasi urban, ini bisa mengindikasikan urbanisasi yang berkelanjutan di negara tersebut. Ini bisa berarti pertumbuhan kota-kota besar atau pergeseran penduduk dari daerah pedesaan ke perkotaan.
    
    3. **Indeks Kesenjangan Gaji (Misalnya: di bidang Construction)**: Jika terjadi penurunan dalam indeks kesenjangan dari tahun ke tahun, ini bisa menunjukkan upaya pemerintah atau kebijakan yang berhasil mengurangi kesenjangan ekonomi di negara tersebut. Sebaliknya, peningkatan yang signifikan mungkin menandakan ketidaksetaraan yang semakin memburuk.
    
    **Actionable Insight:**
    1. **Ekonomi**: Jika terjadi penurunan atau stagnasi dalam GDP, pemerintah dapat mempertimbangkan kebijakan stimulus ekonomi atau reformasi struktural untuk mendorong pertumbuhan.
    
    2. **Urbanisasi**: Jika populasi urban terus meningkat, perlu dipertimbangkan infrastruktur dan layanan perkotaan yang memadai untuk menangani pertumbuhan tersebut, seperti transportasi publik yang efisien, perumahan terjangkau, dan pelayanan kesehatan yang memadai.
    
    3. **Indeks Kesenjangan Gaji (Misalnya: di bidang Construction)**: Jika terjadi peningkatan dalam indeks kesenjangan, pemerintah dapat mempertimbangkan kebijakan redistribusi pendapatan, pendidikan yang lebih merata, atau pelatihan keterampilan untuk memperbaiki ketidaksetaraan ekonomi dan sosial.
    """)

  # Filter data untuk negara yang dipilih
    country_data = data[(data['Country'] == selected_country) & (data['Year'] <= 2021)]  # Saring data hingga tahun 2021

    # Ambil kolom dengan tipe data numerik kecuali kolom 'Year'
    numeric_cols = [col for col in data.select_dtypes(include=['int64', 'float64']).columns if col != 'Year']

    # Membuat plot untuk setiap kolom
    for col in numeric_cols:
        fig = px.line(country_data, x='Year', y=col, title=f"Tingkat {col} di {selected_country} Tiap Tahun")
        fig.update_traces(mode='lines+markers', hovertemplate='Tahun: %{x}<br>Nilai: %{y}')
        fig.update_xaxes(title='Tahun')
        fig.update_yaxes(title=col)
        st.plotly_chart(fig)


def clustering_visualization(data):
    # Load the original data
    original_data = pd.read_csv("data cleaned.csv")

    # Load the clustered labels from the saved file
    with open('hierarchy_clust.pkl', 'rb') as file:
        hierarchical_labels = pickle.load(file)

    # Combine country data with hierarchical cluster labels
    data_with_clusters = pd.concat([original_data['Country'], original_data['Year'], pd.Series(hierarchical_labels, name='hierarchy_cluster')], axis=1)

    # Filter data for European countries
    european_countries = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
        'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Hungary',
        'Italy', 'Latvia', 'Lithuania', 'Malta', 'Netherlands', 
        'Norway', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 
        'Spain', 'Sweden', 'Switzerland']
    data_europe = data_with_clusters[data_with_clusters['Country'].isin(european_countries)]

    # Show dropdown for selecting year
    selected_year = st.selectbox("Pilih Tahun", original_data['Year'].unique())

    # Filter data for selected year
    data_europe_year = data_europe[data_europe['Year'] == selected_year]

    # Create a map visualization of the clustering results for European countries
    fig = px.choropleth(data_europe_year,
                        locations="Country",
                        locationmode='country names',
                        color="hierarchy_cluster",
                        color_continuous_scale=px.colors.sequential.Plasma,  # Choose a color scale
                        hover_name="Country",
                        projection="natural earth",
                        title="Klasterisasi Negara Eropa berdasarkan Gender Pay Gap"
                        )

    # Set geographical scope to Europe
    fig.update_geos(scope="europe")

    # Show the map
    st.plotly_chart(fig)
    st.write('''
            Cluster 0 diisi oleh negara yang memiliki nilai GDP dan Urban_population yang tinggi serta tingkat kesenjangan gaji di bidang Professional_scientific.\n
            Cluster 1 diisi oleh negara yang memiliki tingkat kesenjangan gaji yang paling rendah di antara cluster lainnya yaitu di bidang industry, business, mining, electricity_supply, water_supply, construction, retail trade, transportation, financial, real estate, professional_scientific, administrative, dan education.\n
            Cluster 2 diisi oleh negara yang memiliki tingkat kesenjangan gaji yang paling tinggi di bidang industry, business, mining, electricity_supply, water_supply, construction, retail trade, accommodation, information, financial, real estate, administrative, education, dan human_health
''')


    
def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.selectbox("Go to", ["Business Understanding", "Visualisasi Data", "Clustering"])

    data = load_data()  # Memuat data sebelum digunakan

    if selected_page == "Business Understanding":
        business_understanding()
    elif selected_page == "Visualisasi Data":
        visualize_data(data)
        selected_country = st.sidebar.selectbox("Pilih Negara", data['Country'].unique())
        visualize_data_changes(data, selected_country)
    elif selected_page == "Clustering":
        clustering_visualization(data)

if __name__ == "__main__":
    main()
