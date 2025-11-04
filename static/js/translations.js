// Translation dictionary for Japanese and English
const translations = {
    ja: {
        // Navigation
        app_title: "🔍 PatchCore 訓練アプリ",
        nav_dashboard: "ダッシュボード",
        nav_dataset: "データセット",
        nav_config: "設定",
        nav_train: "訓練",
        nav_inference: "推論"
    },
    
    en: {
        // Navigation
        app_title: "🔍 PatchCore Training",
        nav_dashboard: "Dashboard",
        nav_dataset: "Dataset",
        nav_config: "Configuration",
        nav_train: "Training",
        nav_inference: "Inference"
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