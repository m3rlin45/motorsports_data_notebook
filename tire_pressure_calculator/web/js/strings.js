// UI strings for the web app. Kept byte-for-byte in sync with the .NET
// heads' Core/Localization/strings.json — web/tests/strings.test.mjs fails
// if the two ever drift.

export const STRINGS = {
  en: {
    LanguageLabel: 'Lang:',
    LanguageAuto: 'Auto',
    LanguageEnglish: 'English',
    LanguageJapanese: '日本語',
    ModeLabel: 'Mode:',
    ModeManual: 'Manual',
    ModePrediction: 'Circuit Prediction',
    ModelUnavailable: ' (tire model unavailable on this build)',
    CurrentTemp: 'Current °C',
    TargetTempCorr: 'Target °C (Corr. °C)',
    PredictedHotTemp: 'Predicted hot °C',
    TargetBar: 'Target bar',
    SetCold: 'Set Cold:',
    ResetButton: 'Reset to Defaults',
    Track: 'Track',
    Car: 'Car',
    Condition: 'Condition',
    LapWithinStint: 'Lap within stint',
    Ambient: 'Ambient °C',
    CloudCover: 'Cloud cover %',
    TargetLapTime: 'Target lap time',
    Compound: 'Tire',
    ConditionDry: 'Dry',
    ConditionDamp: 'Damp',
    ConditionWet: 'Wet',
    TempAdjustZero: 'Temp adjust: 0%',
    TempAdjustFormat: 'Temp adjust: {0}%',
    PredictedHotPrefix: 'Predicted hot',
  },
  ja: {
    LanguageLabel: '言語:',
    LanguageAuto: '自動',
    LanguageEnglish: 'English',
    LanguageJapanese: '日本語',
    ModeLabel: 'モード:',
    ModeManual: '手動',
    ModePrediction: '予測モード',
    ModelUnavailable: '（このビルドではタイヤモデルを利用できません）',
    CurrentTemp: '冷間 °C',
    TargetTempCorr: '温間 °C（補正後 °C）',
    PredictedHotTemp: '予測温間 °C',
    TargetBar: '温間 bar',
    SetCold: '冷間設定:',
    ResetButton: 'デフォルトに戻す',
    Track: 'サーキット',
    Car: '車両',
    Condition: 'コンディション',
    LapWithinStint: 'スティント周回',
    Ambient: '外気温 °C',
    CloudCover: '雲量 %',
    TargetLapTime: '目標ラップタイム',
    Compound: 'タイヤ',
    ConditionDry: 'ドライ',
    ConditionDamp: 'ハーフウェット',
    ConditionWet: 'ウェット',
    TempAdjustZero: '温度補正: 0%',
    TempAdjustFormat: '温度補正: {0}%',
    PredictedHotPrefix: '予測温間',
  },
};

export const SUPPORTED_LANGUAGES = ['en', 'ja'];
export const FALLBACK_LANGUAGE = 'en';

// Resolve a user preference ("auto"/"en"/"ja") to an effective language.
export function resolveLanguage(preference, navigatorLanguage) {
  let pref = preference;
  if (!pref || pref === 'auto') pref = (navigatorLanguage || 'en').slice(0, 2).toLowerCase();
  return SUPPORTED_LANGUAGES.includes(pref) ? pref : FALLBACK_LANGUAGE;
}
