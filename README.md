
# Airost-Internship

# AI Inventory Management System  

This project is a real-time inventory management system using AI for object detection and tracking. It is built with YOLOv8, Roboflow, and a custom Tkinter-based GUI.

---

## Features  

- **Object Detection and Tracking**: Detects inventory items in real-time using a live video feed.
- **Check-In/Check-Out**: Updates inventory counts dynamically.
- **Database Integration**: SQLite database to store inventory details.
- **Customizable Interface**: GUI built with Tkinter for ease of use.

---

## Installation  

### Prerequisites  
1. **Python 3.9 or later**  
2. **GPU Support**: Recommended for training models.  

### Steps  

1. Clone this repository: 

       git clone https://github.com/OmarWade3/Airost-Internship.git
       cd Airost-Internship

 2. Install dependencies:
    
        pip install -r requirements.txt
    
 4. Ensure your system has access to a webcam for live video feed.
--- 
## Setup

### API Keys

-   The project uses the **Roboflow API**.
    
-   **Before running**, remove the current API key from the following locations:
    
    -   **Model Training Script (`airost_1_0_(baseline).py`)**:        
        `rf = Roboflow(api_key="YOUR_API_KEY")` 
        
    -   **GUI Code (`app.py`)**:
        `ROBOFLOW_API_KEY = "YOUR_API_KEY"` 
        
-   Replace the API key with your own from [Roboflow](https://roboflow.com).

## Usage

### Training the Model

1.  Open the `airost_1_0_(baseline).py` script in Google Colab.
2.  Update the API key and dataset details as required.
3.  Execute the cells sequentially to train and validate your YOLOv8 model.

### Running the System

1.  Run the GUI application:    

        python app.py
    
3.  Use the interface to:
    -   Start object detection.
    -   Check items in and out.
    -   View and update inventory details.

----------

## Troubleshooting

-   **API Issues**: Ensure your Roboflow API key is valid.
-   **Database Errors**: Verify that the SQLite database file is created and accessible.
-   **Camera Feed**: Check your webcam permissions and ensure no other applications are using it.

----------

## Contributing

Feel free to submit pull requests or report issues to improve this project.

   
