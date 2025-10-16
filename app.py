# -*- coding: utf-8 -*-
"""
FC26 Squad Optimizer - Web GUI
A modern Streamlit interface for squad optimization.
Modern Streamlit arayüzü ile takım optimizasyonu.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os
import sys
import glob
import joblib
from collections import Counter


# Proje modüllerini import et / Import project modules
from texts import TEXTS
from src.data_loader import DataLoader
from src.ml_models import PlayerValuePredictor
from src.genetic_algorithm import GeneticSquadOptimizer
from src.team_synergy_nn import TeamSynergyPredictor

# --- Sayfa Yapılandırması ve Stil / Page Configuration & Styling ---
st.set_page_config(
    page_title="FC26 AI Squad Optimizer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Arayüz için özel CSS stilleri.
# Custom CSS styles for the interface.
st.markdown("""
<style>
    .main-header {
        font-size: 3rem; font-weight: bold; text-align: center;
        color: #1f77b4; margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem; color: #ff7f0e;
        margin-top: 2rem; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Dil ve Metin Yönetimi / Language and Text Management ---

# Tüm arayüz metinleri burada merkezi olarak yönetilir.
# All UI texts are centrally managed here.
# 'TEXTS' sözlüğünün tam ve güncel hali
# 'TEXTS' sözlüğünün tamamını bu blok ile değiştirin.
# Replace the entire 'TEXTS' dictionary with this block.
# 'TEXTS' sözlüğünün tamamını bu blok ile değiştirin.
# Replace the entire 'TEXTS' dictionary with this block.


def get_text(key):
    """Seçili dile göre doğru metni getirir.
       Fetches the correct text for the selected language."""
    # st.session_state'in varlığını garanti altına al / Ensure st.session_state exists
    lang = st.session_state.get('lang', 'tr')
    return TEXTS.get(lang, {}).get(key, f"[{key}]")


# --- Session State ve Model Yükleme / Session State & Model Loading ---

def initialize_session_state():
    """Uygulama boyunca durumu korumak için session state'i başlatır.
       Initializes session state to maintain status across app reruns."""
    defaults = {
        'lang': 'tr',
        'data_loaded': False,
        'loader': None,
        'ml_model': None,
        'synergy_model': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def auto_load_latest_model(model_type):
    """En son eğitilmiş modeli otomatik olarak yükler.
       Automatically loads the most recently trained model."""
    # ... (Bu fonksiyon önceki versiyondaki gibi kalabilir, değişiklik gerekmiyor)
    # This function can remain the same as the previous version, no changes needed.
    model_patterns = {'ml': 'models/trained_models/ml_model_*.pkl', 'synergy': 'models/trained_models/synergy_nn_*.pkl'}
    session_key = f'{model_type}_model'
    if st.session_state.get(session_key) is not None: return
    try:
        model_files = glob.glob(model_patterns[model_type])
        if not model_files: return
        latest_model_path = max(model_files, key=os.path.getctime)
        loaded_data = joblib.load(latest_model_path)
        if isinstance(loaded_data, dict):
            if model_type == 'ml':
                predictor = PlayerValuePredictor()
                predictor.best_model, predictor.scaler, predictor.best_model_name, predictor.results = \
                    loaded_data.get('model'), loaded_data.get('scaler'), loaded_data.get('model_name'), {}
                st.session_state[session_key] = predictor
            elif model_type == 'synergy':
                predictor = TeamSynergyPredictor()
                predictor.model, predictor.scaler, predictor.trained = \
                    loaded_data.get('model'), loaded_data.get('scaler'), loaded_data.get('trained', False)
                st.session_state[session_key] = predictor
        else:
            st.session_state[session_key] = loaded_data
    except Exception:
        pass


# --- Ana Fonksiyonlar / Main Functions ---

def load_data():
    """Veriyi yükler ve session state'i günceller.
       Loads data and updates the session state."""
    csv_path = 'data/players.csv'
    if not os.path.exists(csv_path):
        st.error(get_text("error_file_not_found").format(path=csv_path))
        st.info(get_text("info_save_csv").format(path=csv_path))
        return

    with st.spinner(get_text("spinner_loading_data")):
        loader = DataLoader(csv_path, language=st.session_state.lang)
        loader.load_data()
        loader.clean_data()
        st.session_state.loader = loader
        st.session_state.data_loaded = True

def main():
    """Ana uygulama fonksiyonu, tüm arayüzü yönetir.
       The main app function that controls the entire UI."""
    # Her zaman ilk olarak session state'i başlat / Always initialize session state first
    initialize_session_state()
    
    # Modelleri otomatik yüklemeyi dene / Try to auto-load models
    auto_load_latest_model('ml')
    auto_load_latest_model('synergy')

    st.markdown(f'<h1 class="main-header">{get_text("app_title")}</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
# --- KENAR ÇUBUĞU (SIDEBAR) ---
    with st.sidebar:
        # DİL SEÇİM KUTUSU / LANGUAGE SELECTOR
        lang_options = {'tr': 'Türkçe 🇹🇷', 'en': 'English 🇬🇧'}
        
        current_lang_index = list(lang_options.keys()).index(st.session_state.lang)

        selected_lang_code = st.selectbox(
            label="Dil / Language", 
            options=list(lang_options.keys()),
            format_func=lambda code: lang_options[code],
            index=current_lang_index
        )
        
        if st.session_state.lang != selected_lang_code:
            st.session_state.lang = selected_lang_code
            st.rerun()

        st.title(get_text("nav_title"))
        page_keys = ["home", "data_analysis", "ml_training", "synergy_nn", "optimization", "about"]
        page_names = [get_text(key) for key in page_keys]
        selected_page_name = st.radio(get_text("page_select"), page_names, label_visibility="collapsed")
        page = page_keys[page_names.index(selected_page_name)]

        st.markdown("---")
        if st.session_state.data_loaded:
            st.success(f"✅ {get_text('data_status_loaded')}")
            stats = st.session_state.loader.get_statistics()
            st.metric(get_text("total_players"), f"{stats['total_players']:,}")
            st.metric(get_text("avg_overall"), f"{stats['avg_overall']:.1f}")
        else:
            st.warning(f"⚠️ {get_text('data_status_unloaded')}")
            if st.button(get_text("load_data_button")): 
                load_data()
                st.rerun()

        st.markdown("---")
        st.markdown(f"### {get_text('model_status')}")
        
        # DÜZELTME BURADA / THE FIX IS HERE
        if st.session_state.ml_model is not None:
            st.success(f"✅ {get_text('ml_model')}")
        else:
            st.error(f"❌ {get_text('ml_model')}")
            
        if st.session_state.synergy_model is not None:
            st.success(f"✅ {get_text('synergy_model')}")
        else:
            st.error(f"❌ {get_text('synergy_model')}")

        st.markdown("---")
        st.caption(f"{get_text('version')} 1.3.1 - {get_text('made_with')}")

    # --- SAYFA YÖNLENDİRME / PAGE ROUTING ---
    page_functions = {
        "home": show_home_page, "data_analysis": show_data_analysis_page,
        "ml_training": show_ml_page, "synergy_nn": show_synergy_nn_page,
        "optimization": show_optimization_page, "about": show_about_page
    }
    page_functions[page]()

# --- SAYFA FONKSİYONLARI / PAGE FUNCTIONS ---
# (Aşağıdaki fonksiyonlar, tam dil entegrasyonu ile güncellenmiştir)
# (The functions below are updated with full language integration)

def show_home_page():
    st.markdown(f'<h2 class="sub-header">{get_text("home_quick_start")}</h2>', unsafe_allow_html=True)
    if not st.session_state.data_loaded:
        st.info(get_text("home_start_prompt"))
    else:
        st.success(get_text("home_data_loaded_prompt"))
        stats = st.session_state.loader.get_statistics()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(get_text("total_players"), f"{stats['total_players']:,}")
        c2.metric(get_text("avg_overall"), f"{stats['avg_overall']:.1f}")
        c3.metric(get_text("home_avg_value"), f"€{stats['avg_value']/1e6:.1f}M")
        c4.metric(get_text("home_max_overall"), stats['max_overall'])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### {get_text('home_saved_models')}")
            model_dir = 'models/trained_models'
            if os.path.exists(model_dir) and (models := [f for f in os.listdir(model_dir) if f.endswith('.pkl')]):
                st.write(get_text("home_models_found").format(count=len(models)))
                with st.expander(get_text("home_view_models")):
                    for model in sorted(models, reverse=True)[:5]: st.text(f"📦 {model}")
            else:
                st.info(get_text("home_no_models"))
        with c2:
            st.markdown(f"### {get_text('home_recent_squads')}")
            squad_dir = 'results/best_squads'
            if os.path.exists(squad_dir) and (squads := [f for f in os.listdir(squad_dir) if f.endswith('.csv')]):
                st.write(get_text("home_squads_found").format(count=len(squads)))
                with st.expander(get_text("home_view_squads")):
                    for squad in sorted(squads, reverse=True)[:5]: st.text(f"⚽ {squad}")
            else:
                st.info(get_text("home_no_squads"))

def show_data_analysis_page():
    """Veri analizi sayfasının tüm arayüzünü oluşturur ve gösterir.
       Creates and displays the entire UI for the data analysis page."""
    
    st.markdown(f'<h2 class="sub-header">{get_text("da_title")}</h2>', unsafe_allow_html=True)
    
    # Veri yüklenmediyse, kullanıcıyı uyar ve işlemi durdur.
    # If data isn't loaded, warn the user and stop execution.
    if not st.session_state.data_loaded:
        st.warning(get_text("warning_load_data_first"))
        if st.button(get_text("load_data_button")): 
            load_data()
            st.rerun()
        return

    df = st.session_state.loader.df
    
    # --- Filtreler ---
    with st.expander(get_text("da_filters"), expanded=True):
        c1, c2, c3 = st.columns(3)
        min_overall = c1.slider(get_text("da_min_overall"), 40, 99, 70)
        max_value = c2.number_input(get_text("da_max_value"), min_value=0, value=50000000, step=1000000)
        positions = c3.multiselect(
            get_text("da_position_filter"), 
            options=['GK', 'CB', 'LB', 'RB', 'CDM', 'CM', 'CAM', 'LW', 'RW', 'ST']
        )

    # DataFrame'i filtrelere göre filtrele
    # Filter the DataFrame based on the selected filters
    filtered_df = df[df['overall'] >= min_overall].copy()
    if max_value > 0: 
        filtered_df = filtered_df[filtered_df['value_eur'] <= max_value]
    if positions:
        filtered_df = filtered_df[filtered_df['player_positions'].str.contains('|'.join(positions), na=False)]

    st.info(get_text("da_showing_players").format(count=f"{len(filtered_df):,}", total=f"{len(df):,}"))

    # --- Sekmeler ---
    tab_keys = ["da_tab_distributions", "da_tab_value", "da_tab_nationality", "da_tab_player_list"]
    tab_names = [get_text(key) for key in tab_keys]
    tab1, tab2, tab3, tab4 = st.tabs(tab_names)
    
    # ... (tab1, tab2, tab3 kodları aynı kalır) ...
    with tab1:
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.histogram(filtered_df, x='overall', title=get_text("da_chart_overall_dist")), use_container_width=True)
        if 'age' in filtered_df.columns:
            c2.plotly_chart(px.histogram(filtered_df, x='age', title=get_text("da_chart_age_dist")), use_container_width=True)
    
    with tab2:
        c1, c2 = st.columns(2)
        sample_df = filtered_df.sample(min(1000, len(filtered_df)))
        c1.plotly_chart(px.scatter(sample_df, x='overall', y='value_eur', title=get_text("da_chart_overall_vs_value")), use_container_width=True)
        top_10_valuable = filtered_df.nlargest(10, 'value_eur')
        c2.plotly_chart(px.bar(top_10_valuable, y='short_name', x='value_eur', orientation='h', title=get_text("da_chart_top_10_valuable")), use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            if 'nationality_name' in filtered_df.columns:
                top_nations = filtered_df['nationality_name'].value_counts().nlargest(15)
                fig = px.bar(top_nations, 
                             y=top_nations.index, 
                             x=top_nations.values, 
                             orientation='h', 
                             title=get_text("da_chart_top_15_nations"))
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if 'league_name' in filtered_df.columns:
                top_leagues = filtered_df['league_name'].value_counts().nlargest(10)
                fig = px.pie(values=top_leagues.values, 
                             names=top_leagues.index, 
                             title=get_text("da_chart_top_10_leagues"))
                st.plotly_chart(fig, use_container_width=True)

    # --- Oyuncu Listesi Sekmesi (Güncellendi) ---
    with tab4:
        st.markdown(f"### {get_text('da_tab_player_list')}")
        
        display_cols = ['short_name', 'overall', 'potential', 'age', 'value_eur', 'nationality_name', 'league_name', 'club_name']
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        
        if available_cols:
            # DEĞİŞİKLİK BURADA: .head(100) kaldırıldı.
            # CHANGE IS HERE: .head(100) has been removed.
            display_df = filtered_df[available_cols].copy()
            
            display_df['value_eur'] = display_df['value_eur'].apply(lambda x: f"€{x/1e6:,.2f}M" if x > 0 else "€0")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            # Bilgilendirme mesajı güncellendi. Artık 'ilk 100' demiyor.
            st.info(f"Filtreyle eşleşen toplam {len(display_df):,} oyuncu listeleniyor.")
        else:
            st.warning("Gösterilecek sütun bulunamadı. / No displayable columns found.")

def show_ml_page():
    """Makine öğrenmesi sayfasının tüm arayüzünü oluşturur ve gösterir.
       Creates and displays the entire UI for the machine learning page."""
        
    st.markdown(f'<h2 class="sub-header">{get_text("ml_title")}</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning(get_text("warning_load_data_first"))
        if st.button(get_text("load_data_button")): 
            load_data()
            st.rerun()
        return
    
    loader = st.session_state.loader
    st.info(get_text("ml_info"))
    
    tab_keys = ["ml_tab_train", "ml_tab_compare", "ml_tab_undervalued"]
    tab_names = [get_text(key) for key in tab_keys]
    tab1, tab2, tab3 = st.tabs(tab_names)
    
    # --- MODEL EĞİTİMİ SEKMESİ (DÜZELTİLMİŞ) ---
    with tab1:
        st.markdown(f"### {get_text('ml_train_new')}")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**{get_text('ml_model_features_title')}**")
            st.write(get_text('ml_model_feature_1'))
            st.write(get_text('ml_model_feature_2'))
            st.write(get_text('ml_model_feature_3'))
            st.write(get_text('ml_model_feature_4'))
            st.write(get_text('ml_model_feature_5'))
        
        with col2:
            test_size = st.slider(get_text("ml_test_data_ratio"), 0.1, 0.4, 0.2, 0.05)
            st.metric(get_text("ml_training_data"), f"%{int((1-test_size)*100)}")
            st.metric(get_text("ml_test_data"), f"%{int(test_size*100)}")

        with st.expander(get_text("ml_hyperparameters")):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**{get_text('ml_rf_settings')}**")
                rf_n_estimators = st.number_input(get_text('ml_n_estimators'), 50, 500, 100, 10, key='rf_n')
                rf_max_depth = st.number_input(get_text('ml_max_depth'), 0, 50, 0, 5, key='rf_d')
            with c2:
                st.markdown(f"**{get_text('ml_gb_settings')}**")
                gb_n_estimators = st.number_input(get_text('ml_n_estimators'), 50, 500, 100, 10, key='gb_n')
                gb_max_depth = st.number_input(get_text('ml_max_depth'), 0, 50, 0, 5, key='gb_d')
                gb_learning_rate = st.number_input(get_text('ml_learning_rate'), 0.01, 0.5, 0.1, 0.01, key='gb_lr', format="%.2f")
            with c3:
                st.markdown(f"**{get_text('ml_xgb_settings')}**")
                xgb_n_estimators = st.number_input(get_text('ml_n_estimators'), 50, 500, 100, 10, key='xgb_n')
                xgb_max_depth = st.number_input(get_text('ml_max_depth'), 0, 50, 0, 5, key='xgb_d')
                xgb_learning_rate = st.number_input(get_text('ml_learning_rate'), 0.01, 0.5, 0.1, 0.01, key='xgb_lr', format="%.2f")

        if st.button(get_text("ml_button_start_training"), type="primary"):
            model_params = {
                'Random Forest': {'n_estimators': rf_n_estimators, 'max_depth': rf_max_depth if rf_max_depth > 0 else None},
                'Gradient Boosting': {'n_estimators': gb_n_estimators, 'max_depth': gb_max_depth if gb_max_depth > 0 else None, 'learning_rate': gb_learning_rate},
                'XGBoost': {'n_estimators': xgb_n_estimators, 'max_depth': xgb_max_depth if xgb_max_depth > 0 else None, 'learning_rate': xgb_learning_rate}
            }
            with st.spinner(get_text("ml_spinner_training")):
                try:
                    X, y = loader.get_features_for_ml()
                    ml_predictor = PlayerValuePredictor()
                    ml_predictor.train(X, y, test_size=test_size, model_params=model_params)
                    st.session_state.ml_model = ml_predictor
                    st.success(get_text("ml_success_trained"))
                except Exception as e:
                    st.error(get_text("ml_error_training").format(e=e))
                    if 'ml_model' in st.session_state: # Hata durumunda eski modeli temizle
                        del st.session_state['ml_model']

        # --- DÜZELTME BURADA BAŞLIYOR ---
        # 1. Sonuçları ve kaydet butonunu, sadece model eğitildikten sonra göster.
        #    'st.session_state.ml_model' var mı ve içinde 'results' özelliği var mı diye kontrol et.
        if st.session_state.ml_model is not None and hasattr(st.session_state.ml_model, 'results') and st.session_state.ml_model.results:
            st.markdown("---")
            st.markdown(f"### {get_text('ml_performance')}")
            
            results = st.session_state.ml_model.results
            results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model', 'r2': 'R_Square'})
            best_model_name = results_df.sort_values(by='R_Square', ascending=False)['Model'].iloc[0]
            
            cols = st.columns(len(results_df))
            for i, (idx, row) in enumerate(results_df.iterrows()):
                with cols[i]:
                    header = f"🏆 {row['Model'].upper()}" if row['Model'] == best_model_name else row['Model'].upper()
                    st.markdown(f"**{header}**")
                    st.metric(get_text("ml_r2_score"), f"{row['R_Square']:.4f}")
                    st.metric(get_text("ml_mae"), f"€{row['mae']/1e6:.2f}M")
                    st.metric(get_text("ml_rmse"), f"€{row['rmse']/1e6:.2f}M")
            
            # 2. Kaydetme butonunu artık dışarıya aldık.
            #    Bu buton tıklandığında, session_state'den modeli alıp kaydedecek.
            if st.button(get_text("ml_button_save_model")):
                with st.spinner("Model kaydediliyor..."):
                    path = f'models/trained_models/ml_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
                    # Modeli session_state'den al
                    predictor_to_save = st.session_state.ml_model
                    predictor_to_save.save_model(path)
                    st.success(get_text("ml_success_saved").format(path=path))
        # --- DÜZELTME BURADA BİTİYOR ---

    # --- MODEL KARŞILAŞTIRMA SEKMESİ (Değişiklik yok) ---
    with tab2:
        if st.session_state.ml_model is None or not hasattr(st.session_state.ml_model, 'results') or not st.session_state.ml_model.results:
            st.info(get_text("ml_info_train_first"))
        else:
            results = st.session_state.ml_model.results
            metrics_df = pd.DataFrame({
                'Model': list(results.keys()),
                'R2 Score': [results[k]['r2'] for k in results.keys()],
                'MAE': [results[k]['mae']/1e6 for k in results.keys()]
            })
            
            c1, c2 = st.columns(2)
            with c1:
                fig_r2 = px.bar(metrics_df, x='Model', y='R2 Score', title=get_text('ml_chart_r2_comparison'))
                st.plotly_chart(fig_r2, use_container_width=True)
            with c2:
                fig_mae = px.bar(metrics_df, x='Model', y='MAE', title=get_text('ml_chart_mae_comparison'))
                st.plotly_chart(fig_mae, use_container_width=True)

    # --- DEĞERİ DÜŞÜK OYUNCULAR SEKMESİ (Değişiklik yok) ---
    with tab3:
        if st.session_state.ml_model is None:
            st.info(get_text("ml_info_train_first"))
        else:
            st.markdown(f"### {get_text('ml_tab_undervalued')}")
            
            threshold = st.slider(get_text("ml_undervalued_threshold"), 0.1, 2.0, 0.5, 0.1, help=get_text("ml_undervalued_help"))
            
            if st.button(get_text("ml_button_find_undervalued")):
                with st.spinner(get_text("ml_spinner_finding")):
                    X, y = loader.get_features_for_ml()
                    undervalued = st.session_state.ml_model.find_undervalued_players(X, y, loader.df, threshold)
                    
                    if not undervalued.empty:
                        st.success(get_text("ml_undervalued_found").format(count=len(undervalued)))
                        
                        display_df = undervalued.head(50).copy()
                        
                        for col in ['actual_value', 'predicted_value', 'value_diff']:
                            if col in display_df: display_df[col] = display_df[col].apply(lambda x: f"€{x/1e6:,.2f}M")
                        if 'value_ratio' in display_df: display_df['value_ratio'] = display_df['value_ratio'].apply(lambda x: f"{x:.2f}x")
                        
                        display_df.rename(columns=get_text('ml_undervalued_df_cols'), inplace=True)
                        
                        st.dataframe(display_df, hide_index=True, use_container_width=True)
                        st.info(get_text("ml_undervalued_info_showing").format(count=len(undervalued)))
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric(get_text("ml_undervalued_avg_overall"), f"{undervalued['overall'].mean():.1f}")
                        c2.metric(get_text("ml_undervalued_avg_age"), f"{undervalued['age'].mean():.1f}")
                        c3.metric(get_text("ml_undervalued_avg_potential"), f"{undervalued['potential'].mean():.1f}")
                    else:
                        st.warning(get_text("ml_undervalued_not_found"))


def show_synergy_nn_page():
    """Sinerji NN sayfasının arayüzünü oluşturur (eğitim ve test).
       Creates the UI for the Synergy NN page (training and testing)."""
        
    st.markdown(f'<h2 class="sub-header">{get_text("syn_title")}</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning(get_text("warning_load_data_first"))
        if st.button(get_text("load_data_button")): 
            load_data()
            st.rerun()
        return
    
    loader = st.session_state.loader
    st.info(get_text("syn_info"))
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.session_state.synergy_model:
            st.success(get_text("syn_model_ready"))
            st.info(get_text("syn_model_overwrite_info"))
        else:
            st.warning(get_text("syn_model_not_ready"))
    with col2:
        st.markdown(f"**{get_text('syn_model_details_title')}**")
        st.write(get_text("syn_model_detail_1"))
        st.write(get_text("syn_model_detail_2"))
        st.write(get_text("syn_model_detail_3"))
        st.write(get_text("syn_model_detail_4"))
    
    st.markdown("---")
    
    tab_keys = ["syn_tab_train", "syn_tab_test"]
    tab_names = [get_text(key) for key in tab_keys]
    tab1, tab2 = st.tabs(tab_names)
    
    # --- MODEL EĞİTİMİ SEKMESİ (GÜNCELLENDİ) ---
    with tab1:
        st.markdown(f"### {get_text('syn_train_new')}")
        num_samples = st.number_input(
            get_text("syn_training_samples"), 100, 10000, 1000, 100,
            help=get_text("syn_samples_help")
        )

        # --- YENİ EKLENEN HİPERPARAMETRE BÖLÜMÜ ---
        with st.expander(get_text("syn_hyperparameters")):
            hidden_layers_str = st.text_input(
                get_text("syn_hidden_layers"), 
                "128, 64, 32",
                help=get_text("syn_hidden_layers_help")
            )
            
            c1, c2, c3 = st.columns(3)
            activation = c1.selectbox(get_text("syn_activation"), ['relu', 'tanh', 'logistic'])
            solver = c2.selectbox(get_text("syn_solver"), ['adam', 'sgd', 'lbfgs'])
            max_iter = c3.number_input(get_text("syn_max_iter"), 100, 1000, 200, 50)

        if st.button(get_text("syn_button_start_training"), type="primary", key="train_synergy_button_final"):
            if 'synergy_training_results' in st.session_state:
                del st.session_state['synergy_training_results']
            
            # Kullanıcının girdiği katman yapısını parse et (sayısal hale getir)
            try:
                hidden_layer_sizes = tuple(int(x.strip()) for x in hidden_layers_str.split(',') if x.strip())
            except ValueError:
                st.error("Gizli Katman Yapısı formatı geçersiz. Lütfen '128, 64, 32' gibi sayılar ve virgüller kullanın.")
                st.stop() # Hata varsa işlemi durdur

            # Parametreleri bir sözlükte topla
            nn_params = {
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': activation,
                'solver': solver,
                'max_iter': max_iter
            }

            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                status_text.text(get_text("syn_spinner_creating_predictor"))
                synergy_predictor = TeamSynergyPredictor()
                progress_bar.progress(10)
                
                status_text.text(get_text("syn_spinner_training").format(count=num_samples))
                # GÜNCELLENMİŞ ÇAĞRI: Parametreleri train fonksiyonuna gönder
                results = synergy_predictor.train(loader.df, n_samples=num_samples, nn_params=nn_params)
                progress_bar.progress(80)

                status_text.text(get_text("syn_spinner_saving"))
                path = f'models/trained_models/synergy_nn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
                synergy_predictor.save_model(path)
                st.session_state.synergy_model = synergy_predictor
                
                st.session_state.synergy_training_results = {"results": results, "path": path}
                progress_bar.progress(100)
                status_text.success(get_text("syn_success_trained"))
                
            except Exception as e:
                status_text.error(f"Error: {e}")

        if 'synergy_training_results' in st.session_state and st.session_state.synergy_training_results:
            saved_data = st.session_state.synergy_training_results
            results = saved_data["results"]
            
            st.info(get_text("syn_model_saved_at").format(path=saved_data["path"]))
            st.markdown(f"### {get_text('syn_training_results')}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(get_text("syn_train_mse"), f"{results['train_mse']:.2f}")
            c2.metric(get_text("syn_test_mse"), f"{results['test_mse']:.2f}")
            c3.metric(get_text("syn_train_r2"), f"{results['train_r2']:.4f}")
            c4.metric(get_text("syn_test_r2"), f"{results['test_r2']:.4f}")
            
            if results['test_r2'] >= 0.8: st.success(get_text("syn_perf_excellent"))
            elif results['test_r2'] >= 0.6: st.info(get_text("syn_perf_good"))
            elif results['test_r2'] >= 0.4: st.warning(get_text("syn_perf_fair"))
            else: st.error(get_text("syn_perf_poor"))

    # --- MODEL TESTİ SEKMESİ (Değişiklik yok) ---
    with tab2:
        # Bu sekmede herhangi bir değişiklik gerekmiyor.
        if not st.session_state.synergy_model:
            st.info(get_text("syn_model_not_ready"))
        else:
            # ... (Bu sekmenin geri kalanı aynı kalabilir) ...
            st.info(get_text("syn_test_info"))
            
            temp_opt = GeneticSquadOptimizer(loader.df)
            formation_options = list(temp_opt.formations.keys())
            formation = st.selectbox(get_text("syn_test_formation"), formation_options)
            
            if st.button(get_text("syn_button_test"), key="test_synergy_button_final"):
                if 'synergy_test_results' in st.session_state:
                    del st.session_state['synergy_test_results']
                
                try:
                    position_map = temp_opt.formations[formation]
                    position_counts = Counter(position_map)
                    
                    test_squad = []
                    used_player_ids = set()
                    
                    for pos, count in position_counts.items():
                        base_pos = pos.replace('L', '').replace('R', '')
                        
                        eligible = loader.df[
                            (loader.df['player_positions'].str.contains(base_pos, na=False)) &
                            (~loader.df['player_id'].isin(used_player_ids))
                        ]
                        
                        if len(eligible) < count:
                            st.error(f"Test takımı oluşturulamadı. Yeterli '{base_pos}' oyuncusu bulunamadı (İhtiyaç: {count}, Bulunan: {len(eligible)}).")
                            test_squad = [] 
                            break 
                        
                        selected_players = eligible.sample(count)
                        
                        for _, player_row in selected_players.iterrows():
                            test_squad.append(player_row)
                            used_player_ids.add(player_row['player_id'])

                    if len(test_squad) == 11:
                        squad_df = pd.DataFrame(test_squad)
                        final_squad_sorted = [row for _, row in squad_df.iterrows()]

                        synergy_score = st.session_state.synergy_model.predict_synergy(final_squad_sorted, position_map)
                        
                        st.session_state.synergy_test_results = {
                            "score": synergy_score,
                            "squad": squad_df
                        }

                except Exception as e:
                    st.error(f"Bir hata oluştu: {e}")

            if 'synergy_test_results' in st.session_state and st.session_state.synergy_test_results:
                test_results = st.session_state.synergy_test_results
                squad_df = test_results["squad"]
                
                st.markdown(f"### {get_text('syn_test_results_title')}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(get_text("syn_test_score"), f"{test_results['score']:.1f}/100")
                c2.metric(get_text("syn_test_avg_overall"), f"{squad_df['overall'].mean():.1f}")
                c3.metric(get_text("syn_test_avg_age"), f"{squad_df['age'].mean():.1f}")
                c4.metric(get_text("syn_test_total_value"), f"€{squad_df['value_eur'].sum()/1e6:.1f}M")
                
                st.markdown(f"**{get_text('syn_test_squad_list')}**")
                st.dataframe(squad_df[['short_name', 'overall', 'age', 'club_name', 'nationality_name']], hide_index=True, use_container_width=True)

def show_optimization_page():
    """Takım optimizasyon sayfasının tüm arayüzünü oluşturur ve gösterir.
       Creates and displays the entire UI for the team optimization page."""

    st.markdown(f'<h2 class="sub-header">{get_text("opt_title")}</h2>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning(get_text("warning_load_data_first"))
        if st.button(get_text("load_data_button")):
            load_data()
            st.rerun()
        return

    loader = st.session_state.loader
    st.info(get_text("opt_info"))

    # Temel Parametreler
    with st.expander(get_text("opt_params"), expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            budget = st.number_input(get_text("opt_budget"), 1000000, 1000000000, 50000000, 1000000, format="%d")
        with col2:
            temp_opt = GeneticSquadOptimizer(loader.df)
            formation = st.selectbox(get_text("opt_formation"), options=list(temp_opt.formations.keys()))

    # Genetik Algoritma Parametreleri
    with st.expander(get_text("opt_ga_params")):
        col1, col2 = st.columns(2)
        generations = col1.slider(get_text("opt_generations"), 10, 100, 30, 5, help=get_text("opt_generations_help"))
        population_size = col2.slider(get_text("opt_population"), 20, 200, 50, 10, help=get_text("opt_population_help"))

    # === YENİ EKLENEN BÖLÜM: GELİŞMİŞ SEÇENEKLER ===
    with st.expander(get_text("opt_advanced_options")):
        col1, col2 = st.columns(2)
        
        with col1:
            use_ml = st.checkbox(
                get_text("opt_use_ml"),
                value=False,
                help=get_text("opt_use_ml_help"),
                disabled=st.session_state.ml_model is None
            )
            if st.session_state.ml_model is None:
                st.warning(get_text("opt_ml_model_not_found"))
        
        with col2:
            use_synergy = st.checkbox(
                get_text("opt_use_synergy"),
                value=False,
                help=get_text("opt_use_synergy_help"),
                disabled=st.session_state.synergy_model is None
            )
            if st.session_state.synergy_model is None:
                st.warning(get_text("opt_synergy_model_not_found"))

    # Optimizasyonu Başlatma
    if st.button(get_text("opt_button_start"), type="primary", key="start_optimization"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current_gen, total_gen, best_fitness):
            progress = int((current_gen / total_gen) * 100)
            progress_bar.progress(progress)
            status_text.text(get_text("opt_status_generation").format(curr=current_gen, total=total_gen, fit=best_fitness))

        try:
            with st.spinner(get_text("opt_spinner_optimizing")):
                optimizer = GeneticSquadOptimizer(loader.df, formation)
                
                # === GÜNCELLENMİŞ OPTIMIZE ÇAĞRISI ===
                result = optimizer.optimize(
                    budget=budget,
                    population_size=population_size,
                    generations=generations,
                    elite_size=5,
                    use_ml=use_ml,
                    ml_predictor=st.session_state.ml_model if use_ml else None,
                    use_synergy=use_synergy,
                    synergy_predictor=st.session_state.synergy_model if use_synergy else None,
                    progress_callback=update_progress
                )
            
            progress_bar.progress(100)
            status_text.success(get_text("opt_success_completed"))
            
            # Sonuçları 'session_state'e kaydet
            st.session_state.optimization_result = result

        except Exception as e:
            st.error(get_text("opt_error").format(e=e))
            st.info(get_text("opt_error_suggestion"))
            if 'optimization_result' in st.session_state:
                del st.session_state['optimization_result']
    
    # Sonuçları 'session_state'den okuyarak göster
    if 'optimization_result' in st.session_state:
        result = st.session_state.optimization_result
        
        st.markdown(f"### {get_text('opt_results_title')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(get_text("opt_total_fitness"), f"{result['fitness']:.1f}")
        c2.metric(get_text("opt_total_value"), f"€{result['cost']/1e6:.1f}M")
        c3.metric(get_text("opt_avg_overall_squad"), f"{result['avg_overall']:.1f}")
        c4.metric(get_text("opt_chemistry"), f"{result['chemistry']:.1f}")
        
        st.markdown(f"**{get_text('opt_squad_list')}**")
        squad_df_data = [{'Position': pos, 'Name': p['short_name'], 'Overall': p['overall'], 'Value': f"€{p['value_eur']/1e6:.2f}M", 'Club': p['club_name']} for p, pos in zip(result['squad'], result['positions'])]
        st.dataframe(pd.DataFrame(squad_df_data), use_container_width=True, hide_index=True)

        if result.get('bench'):
            st.markdown(f"**{get_text('opt_bench_list')}**")
            bench_df_data = [{'Name': p['short_name'], 'Overall': p['overall'], 'Value': f"€{p['value_eur']/1e6:.2f}M", 'Positions': p['player_positions']} for p in result['bench']]
            st.dataframe(pd.DataFrame(bench_df_data), use_container_width=True, hide_index=True)

        if st.button(get_text("opt_button_save_squad"), key="save_squad"):
            path = f'results/best_squads/squad_{formation}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Kaydetmek için yeni bir optimizer nesnesi oluşturmak daha güvenli olabilir
            temp_optimizer = GeneticSquadOptimizer(loader.df, formation)
            temp_optimizer.export_squad(result, path)
            st.success(get_text("opt_success_squad_saved").format(path=path))

def show_about_page():
    """Hakkında sayfasının arayüzünü oluşturur.
       Creates the UI for the about page."""
    st.markdown(f'<h2 class="sub-header">{get_text("about_title")}</h2>', unsafe_allow_html=True)
    st.markdown(get_text("about_subtitle"))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {get_text('about_features_title')}")
        st.markdown(get_text("about_features_list"))
    with col2:
        st.markdown(f"### {get_text('about_tech_title')}")
        st.markdown(get_text("about_tech_list"))

# --- Uygulamayı Başlat / Run the App ---
if __name__ == "__main__":
    main()