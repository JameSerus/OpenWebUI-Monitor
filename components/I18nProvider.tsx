"use client";

import { PropsWithChildren, useEffect } from "react";
import i18next from "i18next";
import { I18nextProvider, initReactI18next } from "react-i18next";
import LanguageDetector from "i18next-browser-languagedetector";

import enCommon from "@/locales/en/common.json";
import deCommon from "@/locales/de/common.json";

const i18n = i18next
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      en: {
        common: enCommon,
      },
      de: {
        common: deCommon,
      },
    },
    fallbackLng: "de",
    interpolation: {
      escapeValue: false,
    },
  });

export default function I18nProvider({ children }: PropsWithChildren) {
  return <I18nextProvider i18n={i18next}>{children}</I18nextProvider>;
}
