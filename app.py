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
    st.markdown('''
    <div style="text-align: justify;">
    <p><strong>1. Pola Distribusi Data:</strong></p>
    <p>Histogram di atas memberikan gambaran tentang pola distribusi data pada setiap kolom numerik. Distribusi data ini menggambarkan sebaran nilai-nilai di dalam setiap kolom dan memberikan wawasan tentang bagaimana data tersebar.</p>
    
    <p><strong>2. Puncak Histogram:</strong></p>
    <p>Puncak histogram menunjukkan di mana sebagian besar nilai-nilai data berada. Puncak yang tinggi menunjukkan bahwa terdapat konsentrasi nilai-nilai di sekitar titik tersebut. Semakin tinggi puncaknya, semakin besar konsentrasi nilai-nilai di wilayah tersebut.</p>
    
    <p><strong>3. Lebar Histogram:</strong></p>
    <p>Lebar histogram mencerminkan rentang nilai-nilai yang mungkin ada dalam kolom tersebut. Semakin lebar histogram, semakin besar rentang nilai-nilai yang tercakup dalam data. Lebar yang lebih besar menunjukkan variasi yang lebih besar dalam data, sedangkan lebar yang lebih kecil menunjukkan konsentrasi nilai-nilai di rentang nilai yang lebih sempit.</p>

    <p>Dengan memahami pola distribusi data melalui histogram ini, kita dapat mengidentifikasi titik-titik penting seperti pusat konsentrasi nilai (puncak histogram) dan seberapa luas variasi nilai-nilai tersebut (lebar histogram). Informasi ini dapat membantu dalam analisis lebih lanjut dan pengambilan keputusan yang tepat berdasarkan karakteristik data yang diamati.</p>
    </div>
    ''', unsafe_allow_html=True)




def visualize_data_changes(data, selected_country):
    st.title(f"Data untuk Negara {selected_country} dari 2010-2021")
    st.markdown("""
    <div style="text-align: justify;">
    <p><strong>Interpretasi:</strong></p>
    <p>Visualisasi ini memperlihatkan perubahan nilai numerik di setiap kolom selama periode 2010-2021 untuk negara yang dipilih. Setiap garis pada grafik mewakili tren perubahan nilai dari satu kolom tertentu dari tahun ke tahun.</p>
    
    <p><strong>Insight:</strong></p>
    <ol>
    <li><strong>GDP (PDB):</strong> Jika terdapat peningkatan yang signifikan dalam grafik GDP dari tahun ke tahun, ini menunjukkan pertumbuhan ekonomi yang kuat di negara tersebut. Sebaliknya, penurunan atau stagnasi bisa menjadi sinyal adanya masalah ekonomi yang perlu ditangani.</li>
    
    <li><strong>Urban Population (Populasi Urban):</strong> Jika grafik menunjukkan peningkatan yang stabil dalam populasi urban, ini bisa mengindikasikan urbanisasi yang berkelanjutan di negara tersebut. Ini bisa berarti pertumbuhan kota-kota besar atau pergeseran penduduk dari daerah pedesaan ke perkotaan.</li>
    
    <li><strong>Indeks Kesenjangan Gaji (Misalnya: di bidang Construction):</strong> Jika terjadi penurunan dalam indeks kesenjangan dari tahun ke tahun, ini bisa menunjukkan upaya pemerintah atau kebijakan yang berhasil mengurangi kesenjangan ekonomi di negara tersebut. Sebaliknya, peningkatan yang signifikan mungkin menandakan ketidaksetaraan yang semakin memburuk.</li>
    </ol>
    
    <p><strong>Actionable Insight:</strong></p>
    <ol>
    <li><strong>Ekonomi:</strong> Jika terjadi penurunan atau stagnasi dalam GDP, pemerintah dapat mempertimbangkan kebijakan stimulus ekonomi atau reformasi struktural untuk mendorong pertumbuhan.</li>
    
    <li><strong>Urbanisasi:</strong> Jika populasi urban terus meningkat, perlu dipertimbangkan infrastruktur dan layanan perkotaan yang memadai untuk menangani pertumbuhan tersebut, seperti transportasi publik yang efisien, perumahan terjangkau, dan pelayanan kesehatan yang memadai.</li>
    
    <li><strong>Indeks Kesenjangan Gaji (Misalnya: di bidang Construction):</strong> Jika terjadi peningkatan dalam indeks kesenjangan, pemerintah dapat mempertimbangkan kebijakan redistribusi pendapatan, pendidikan yang lebih merata, atau pelatihan keterampilan untuk memperbaiki ketidaksetaraan ekonomi dan sosial.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)


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


def ciri_ciri_cluster(data):
    for cluster_id in range(3):  # assuming there are 3 clusters
        with st.expander(f"Ciri-Ciri Cluster {cluster_id}"):
            st.markdown(f"Ciri-Ciri Cluster {cluster_id}:")
            if cluster_id == 0:
                st.markdown("""
                - GDP rata-rata sekitar 36%
                - Urban Population sekitar 80%
                - Tingkat kesenjangan di bidang Industry sekitar 14%
                - Tingkat kesenjangan di bidang Business sekitar 17%
                - Tingkat kesenjangan di bidang Mining sekitar 10%
                - Tingkat kesenjangan di bidang Manufacturing sekitar 16%
                - Tingkat kesenjangan di bidang Electricity Supply sekitar 13%
                - Tingkat kesenjangan di bidang Water Supply sekitar 3%
                - Tingkat kesenjangan di bidang Construction sekitar 3%
                - Tingkat kesenjangan di bidang Retail Trade sekitar 20%
                - Tingkat kesenjangan di bidang Transportation sekitar 8%
                - Tingkat kesenjangan di bidang Accomodation Supply sekitar 9%
                - Tingkat kesenjangan di bidang Information sekitar 16%
                - Tingkat kesenjangan di bidang Financial sekitar 25.70%
                - Tingkat kesenjangan di bidang Real Estate sekitar 14%
                - Tingkat kesenjangan di bidang Professional Scientific sekitar 22%
                - Tingkat kesenjangan di bidang Administrative sekitar 10% 
                - Tingkat kesenjangan di bidang Public Administration sekitar 10%
                - Tingkat kesenjangan di bidang Education sekitar 10.94%
                - Tingkat kesenjangan di bidang Human Health sekitar 17%
                - Tingkat kesenjangan di bidang Arts sekitar 18%
                - Tingkat kesenjangan di bidang lainnya sekitar 17%
                """)
            elif cluster_id == 1:
                st.markdown("""
                - GDP rata-rata sekitar 12.120%)
                - Urban Population sekitar 60.64%
                - Tingkat kesenjangan di bidang Industry sekitar 8.95%
                - Tingkat kesenjangan di bidang Business sekitar 12.90%
                - Tingkat kesenjangan di bidang Mining sekitar 4.11%
                - Tingkat kesenjangan di bidang Manufacturing sekitar 21.05%
                - Tingkat kesenjangan di bidang Electricity Supply sekitar 4.57%
                - Tingkat kesenjangan di bidang Water Supply sekitar -4.28%
                - Tingkat kesenjangan di bidang Construction sekitar -15.79%
                - Tingkat kesenjangan di bidang Retail Trade sekitar 17.46%
                - Tingkat kesenjangan di bidang Transportation sekitar -5.38%
                - Tingkat kesenjangan di bidang Accommodation Supply sekitar 11.00%
                - Tingkat kesenjangan di bidang Information sekitar 17.75%
                - Tingkat kesenjangan di bidang Financial sekitar 25.19%
                - Tingkat kesenjangan di bidang Real Estate sekitar 7.10%
                - Tingkat kesenjangan di bidang Professional Scientific sekitar 11.47%
                - Tingkat kesenjangan di bidang Administrative sekitar -3.15%
                - Tingkat kesenjangan di bidang Public Administration sekitar 8.99%
                - Tingkat kesenjangan di bidang Education sekitar 10.82%
                - Tingkat kesenjangan di bidang Human Health sekitar 20.12%
                - Tingkat kesenjangan di bidang Arts sekitar 18.47%
                - Tingkat kesenjangan di bidang lainnya sekitar 19.53%
                """)
            elif cluster_id == 2:
                st.markdown("""
                - GDP rata-rata sekitar 13,177.92
                - Urban Population sekitar 67.03%
                - Tingkat kesenjangan di bidang Industry sekitar 18.76%
                - Tingkat kesenjangan di bidang Business sekitar 18.54%
                - Tingkat kesenjangan di bidang Mining sekitar 13.15%
                - Tingkat kesenjangan di bidang Manufacturing sekitar 25.26%
                - Tingkat kesenjangan di bidang Electricity Supply sekitar 14.24%
                - Tingkat kesenjangan di bidang Water Supply sekitar 6.83%
                - Tingkat kesenjangan di bidang Construction sekitar 4.41%
                - Tingkat kesenjangan di bidang Retail Trade sekitar 25.16%
                - Tingkat kesenjangan di bidang Transportation sekitar 2.77%
                - Tingkat kesenjangan di bidang Accommodation Supply sekitar 13.86%
                - Tingkat kesenjangan di bidang Information sekitar 26.97%
                - Tingkat kesenjangan di bidang Financial sekitar 36.51%
                - Tingkat kesenjangan di bidang Real Estate sekitar 15.18%
                - Tingkat kesenjangan di bidang Professional Scientific sekitar 18.60%
                - Tingkat kesenjangan di bidang Administrative sekitar 11.73%
                - Tingkat kesenjangan di bidang Public Administration sekitar 8.23%
                - Tingkat kesenjangan di bidang Education sekitar 13.61%
                - Tingkat kesenjangan di bidang Human Health sekitar 22.92%
                - Tingkat kesenjangan di bidang Arts sekitar 16.67%
                - Tingkat kesenjangan di bidang lainnya sekitar 13.36%
                """)


    st.write('''
            Cluster 0 mewakili negara atau wilayah dengan nilai GDP dan Urban_population yang tinggi serta tingkat kesenjangan gaji di bidang Professional_scientific.\n
            Cluster 1 mewakili negara atau wilayah dengan tingkat kesenjangan gaji yang paling rendah di antara cluster lainnya yaitu pada bidang industry, business, mining, electricity_supply, water_supply, construction, retail trade, transportation, financial, real estate, professional_scientific, administrative, dan education.\n
            Cluster 2 mewakili negara atau wilayah dengan tingkat kesenjangan gaji yang paling tinggi di bidang industry, business, mining, electricity_supply, water_supply, construction, retail trade, accommodation, information, financial, real estate, administrative, education, dan human_health
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
        ciri_ciri_cluster(data)


if __name__ == "__main__":
    main()
