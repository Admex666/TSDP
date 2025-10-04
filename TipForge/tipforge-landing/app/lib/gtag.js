export const GA_MEASUREMENT_ID = "G-1ZPGGY3XLV"

// Page view esemény
export const pageview = (url) => {
  window.gtag("config", GA_MEASUREMENT_ID, {
    page_path: url,
  });
};
