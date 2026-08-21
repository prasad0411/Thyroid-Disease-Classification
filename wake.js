const { chromium } = require('playwright');

const urls = [
  'https://thyroid-disease-classification.streamlit.app/',
  'https://job-scraping-analytics.streamlit.app/'
];

(async () => {
  const browser = await chromium.launch();
  let failed = false;

  for (const url of urls) {
    const page = await browser.newPage();
    try {
      console.log('Visiting:', url);
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
      await page.waitForTimeout(5000);

      const wakeBtn = page.getByRole('button', { name: /get this app back up/i });
      if (await wakeBtn.count() > 0) {
        console.log('App sleeping, clicking wake button...');
        await wakeBtn.first().click();
        await page.waitForTimeout(45000);
      } else {
        console.log('App already awake.');
      }
      console.log('Done:', url);
    } catch (err) {
      failed = true;
      console.error('Failed for', url, '->', err.message);
    } finally {
      await page.close();
    }
  }

  await browser.close();
  if (failed) process.exit(1);
})();
