from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

# Define the initial dictionary of URLs to scrape
toc_links = {
    "Allergy and Immunology": "https://sniv3r2.github.io/d/toc.htm?path=allergy-and-immunology",
    "Anesthesiology": "https://sniv3r2.github.io/d/toc.htm?path=anesthesiology",
    "Cardiovascular Medicine": "https://sniv3r2.github.io/d/toc.htm?path=cardiovascular-medicine",
    "Dermatology": "https://sniv3r2.github.io/d/toc.htm?path=dermatology",
    "Drug Information": "https://sniv3r2.github.io/d/toc.htm?path=drug-information",
    "Emergency Medicine (Adult and Pediatric)": "https://sniv3r2.github.io/d/toc.htm?path=emergency-medicine-adult-and-pediatric",
    "Endocrinology and Diabetes": "https://sniv3r2.github.io/d/toc.htm?path=endocrinology-and-diabetes",
    "Family Medicine and General Practice": "https://sniv3r2.github.io/d/toc.htm?path=family-medicine-and-general-practice",
    "Gastroenterology and Hepatology": "https://sniv3r2.github.io/d/toc.htm?path=gastroenterology-and-hepatology",
    "General Surgery": "https://sniv3r2.github.io/d/toc.htm?path=general-surgery",
    "Geriatrics": "https://sniv3r2.github.io/d/toc.htm?path=geriatrics",
    "Hematology": "https://sniv3r2.github.io/d/toc.htm?path=hematology",
    "Hospital Medicine": "https://sniv3r2.github.io/d/toc.htm?path=hospital-medicine",
    "Infectious Diseases": "https://sniv3r2.github.io/d/toc.htm?path=infectious-diseases",
    "Lab Interpretation™": "https://sniv3r2.github.io/d/toc.htm?path=lab-interpretation",
    "Nephrology and Hypertension": "https://sniv3r2.github.io/d/toc.htm?path=nephrology-and-hypertension",
    "Neurology": "https://sniv3r2.github.io/d/toc.htm?path=neurology",
    "Obstetrics, Gynecology and Women's Health": "https://sniv3r2.github.io/d/toc.htm?path=obstetrics-gynecology-and-womens-health",
    "Oncology": "https://sniv3r2.github.io/d/toc.htm?path=oncology",
    "Palliative Care": "https://sniv3r2.github.io/d/toc.htm?path=palliative-care",
    "Pediatrics": "https://sniv3r2.github.io/d/toc.htm?path=pediatrics",
    "Primary Care (Adult)": "https://sniv3r2.github.io/d/toc.htm?path=primary-care-adult",
    "Primary Care Sports Medicine (Adolescents and Adults)": "https://sniv3r2.github.io/d/toc.htm?path=primary-care-sports-medicine-adolescents-and-adults",
    "Psychiatry": "https://sniv3r2.github.io/d/toc.htm?path=psychiatry",
    "Pulmonary and Critical Care Medicine": "https://sniv3r2.github.io/d/toc.htm?path=pulmonary-and-critical-care-medicine",
    "Rheumatology": "https://sniv3r2.github.io/d/toc.htm?path=rheumatology",
    "Sleep Medicine": "https://sniv3r2.github.io/d/toc.htm?path=sleep-medicine"
}
# Configure Selenium
options = Options()
options.headless = True  # Run in headless mode (no browser window)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Function to determine if a URL is a content page based on its structure
def is_content_page(url):
    return "topic.htm" in url

# Function to scrape links from a specific URL
def extract_links_from_page(url):
    try:
        driver.get(url)
        # Wait until the pageContent div is loaded
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "pageContent")))
        page_content = driver.find_element(By.ID, "pageContent")
        
        # Find all the anchor elements within pageContent
        links = page_content.find_elements(By.TAG_NAME, "a")
        urls = [(link.text, link.get_attribute("href")) for link in links if link.get_attribute("href")]
        return urls
    except TimeoutException:
        print(f"Timeout while trying to access {url}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Recursive function to handle nested scraping and track the click path
def recursive_scrape(category_url, clicked_path=None, depth=2):
    if clicked_path is None:
        clicked_path = []
    if depth == 0:
        return []

    # Extract links from the current page
    top_level_links = extract_links_from_page(category_url)
    all_content_links = []

    # Recursively navigate through each link found on the page
    for link_text, link_url in top_level_links:
        if link_url and link_url not in [link['url'] for link in all_content_links]:
            # Track the navigation path
            new_path = clicked_path + [link_text]
            if is_content_page(link_url):
                # If it's a content page, save it
                all_content_links.append({"url": link_url, "path": new_path})
            else:
                # Otherwise, keep scraping deeper
                deeper_links = recursive_scrape(link_url, new_path, depth - 1)
                all_content_links.extend(deeper_links)
    
    return all_content_links

# Main function to run the scraping for all categories
def scrape_all_categories(toc_links):
    all_scraped_data = {}
    for category, url in toc_links.items():
        print(f"Scraping category: {category}...")
        category_content = recursive_scrape(url)
        all_scraped_data[category] = category_content
        print(f"Found {len(category_content)} content links in {category}.\n")
    return all_scraped_data

# Run the scraper
scraped_data = scrape_all_categories(toc_links)

# Save the result to a JSON file for later use
with open("categorized_content_links.json", "w") as file:
    json.dump(scraped_data, file, indent=4)

# Display summary of scraped data
for category, content in scraped_data.items():
    print(f"Category: {category} ({len(content)} content links)")
    for res in content:
        print(f" - URL: {res['url']} | Path: {' > '.join(res['path'])}")

# Close the browser
driver.quit()
