// Translation dictionary for Japanese and English
const translations = {
    ja: {
        // Navigation
        app_title: "🔍 PatchCore 訓練アプリ",
        nav_dashboard: "ダッシュボード",
        nav_upload: "アップロード",
        nav_dataset: "データセット",
        nav_config: "設定",
        nav_train: "訓練",
        nav_inference: "推論",

        // Upload Page
        upload_title: "📤 訓練画像をアップロード",
        upload_dataset_label: "データセット:",
        upload_select_dataset: "-- データセットを選択または新規作成 --",
        upload_create_new: "➕ 新規作成",
        upload_refresh: "🔄 更新",
        upload_normal_images: "正常画像",
        upload_abnormal_images: "異常画像",
        upload_total_images: "合計画像数",
        upload_dataset_path: "データセットパス:",
        upload_normal_title: "✅ 正常画像をアップロード",
        upload_abnormal_title: "❌ 異常画像をアップロード",
        upload_select_files: "画像を選択:",
        upload_button_normal: "⬆️ 正常画像をアップロード",
        upload_button_abnormal: "⬆️ 異常画像をアップロード",
        upload_normal_preview: "正常画像:",
        upload_abnormal_preview: "異常画像:",
        upload_quick_actions: "⚡ クイックアクション",
        upload_goto_config: "➡️ このデータセットの設定を生成",
        upload_goto_train: "🚀 訓練へ移動",
        upload_delete_dataset: "🗑️ このデータセットを削除",
        upload_create_dataset_title: "新しいデータセットを作成",
        upload_dataset_name: "データセット名:",
        upload_dataset_name_hint: "小文字、数字、アンダースコアのみ使用可能",
        upload_cancel: "キャンセル",
        upload_create: "作成"
    },

    en: {
        // Navigation
        app_title: "🔍 PatchCore Training",
        nav_dashboard: "Dashboard",
        nav_upload: "Upload",
        nav_dataset: "Dataset",
        nav_config: "Configuration",
        nav_train: "Training",
        nav_inference: "Inference",

        // Upload Page
        upload_title: "📤 Upload Training Images",
        upload_dataset_label: "Dataset:",
        upload_select_dataset: "-- Select or Create New Dataset --",
        upload_create_new: "➕ Create New",
        upload_refresh: "🔄 Refresh",
        upload_normal_images: "Normal Images",
        upload_abnormal_images: "Abnormal Images",
        upload_total_images: "Total Images",
        upload_dataset_path: "Dataset Path:",
        upload_normal_title: "✅ Upload Normal Images",
        upload_abnormal_title: "❌ Upload Abnormal Images",
        upload_select_files: "Select Images:",
        upload_button_normal: "⬆️ Upload Normal Images",
        upload_button_abnormal: "⬆️ Upload Abnormal Images",
        upload_normal_preview: "Normal Images:",
        upload_abnormal_preview: "Abnormal Images:",
        upload_quick_actions: "⚡ Quick Actions",
        upload_goto_config: "➡️ Generate Config for This Dataset",
        upload_goto_train: "🚀 Go to Training",
        upload_delete_dataset: "🗑️ Delete This Dataset",
        upload_create_dataset_title: "Create New Dataset",
        upload_dataset_name: "Dataset Name:",
        upload_dataset_name_hint: "Use lowercase letters, numbers, and underscores only",
        upload_cancel: "Cancel",
        upload_create: "Create"
    }
};

// Current language (default to Japanese)
let currentLanguage = localStorage.getItem('language') || 'ja';

// Function to get translation
function t(key) {
    return translations[currentLanguage][key] || translations['en'][key] || key;
}

// Function to switch language
function switchLanguage(lang) {
    currentLanguage = lang;
    localStorage.setItem('language', lang);
    
    // Update all elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        const translation = translations[lang][key];
        
        if (translation) {
            if (element.tagName === 'INPUT' && element.type !== 'submit' && element.type !== 'button') {
                if (element.placeholder !== undefined) {
                    element.placeholder = translation;
                }
            } else {
                element.textContent = translation;
            }
        }
    });
    
    // Update language button text
    const langText = document.getElementById('currentLangText');
    if (langText) {
        langText.textContent = lang === 'ja' ? '日本語' : 'English';
    }
    
    // Update dropdown active state
    document.querySelectorAll('.lang-option').forEach(option => {
        option.classList.remove('active');
        if ((lang === 'ja' && option.textContent.includes('日本語')) ||
            (lang === 'en' && option.textContent.includes('English'))) {
            option.classList.add('active');
        }
    });
}

// Initialize language on page load
document.addEventListener('DOMContentLoaded', function() {
    switchLanguage(currentLanguage);
});