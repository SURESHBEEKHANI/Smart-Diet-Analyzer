# Smart Diet Analyzer

Smart Diet Analyzer is an AI-powered application that analyzes food images to provide nutritional information and calorie estimates. The application uses advanced image analysis capabilities to identify food items and estimate their calorie content.

## Features

- Upload food images for analysis
- AI-powered food identification and calorie estimation
- Generate and download a PDF report of the nutritional analysis
- User-friendly interface with Streamlit

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/SURESHBEEKHANI/Smart-Diet-Analyzer.git
    cd Smart-Diet-Analyzer
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the project root directory and add your Groq API key:
    ```env
    GROQ_API_KEY=your_api_key_here
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload a food image using the sidebar and click "Analyze Meal 🍽️" to get the nutritional analysis.

4. Download the generated PDF report or clear the analysis to start over.

## Project Structure

- `app.py`: Main application file containing the Streamlit interface and utility functions.
- `requirements.txt`: List of dependencies required for the project.
- `.env`: Environment file to store sensitive information like API keys.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Share on LinkedIn

Excited to share my latest project, Smart Diet Analyzer! 🎉

Smart Diet Analyzer is an AI-powered application that analyzes food images to provide nutritional information and calorie estimates. Whether you're a fitness enthusiast or just curious about your meals, this tool can help you make informed dietary choices.

Check out the project on GitHub: [Smart Diet Analyzer](https://github.com/SURESHBEEKHANI/Smart-Diet-Analyzer)
