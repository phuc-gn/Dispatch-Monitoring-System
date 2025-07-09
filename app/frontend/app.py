import streamlit as st

import os, requests


API_URL = 'http://backend:8000/process_video'
SHARED_DIR = '/shared_volume'

st.title('Dispatch Monitoring Demo')

# List all .mp4 files in the shared directory
available_videos = [f.name for f in os.scandir(SHARED_DIR) if f.is_file() and f.name.endswith('.mp4')]

# Dropdown to choose existing video
selected_video = st.selectbox('Select a video file from shared volume', available_videos)

# Input for processing duration in minutes
minutes = st.number_input('Processing duration (minutes)', value=1)

st.write('Please nagivate to the mounted shared volume `app/shared_volume` to see the output video. Sorry for the inconvenience due to the limitations of the current setup (codec issues with Streamlit).')

if st.button('Start processing') and selected_video:
    with st.spinner('Processing...'):
        response = requests.post(API_URL, json={'filename': selected_video, 'minutes': minutes})

    if response.status_code == 200:
        st.success(f"Done! Please check the {response.json().get('output', 'output video')} in the shared volume.")
        st.video(os.path.join(SHARED_DIR, response.json()['output']))
        
    else:
        st.error(response.json().get('error', 'Unknown error'))