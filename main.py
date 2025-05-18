import streamlit as st
import cv2
import numpy as np
# from PIL import Image
# import pickle
import os
import time
import io

import google.generativeai as genai
from inference_sdk import InferenceHTTPClient

# --- PAGE CONFIG MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="የስሜት መመርመሪያ እና AI Chat", layout="wide")

# --- Constants ---
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
AMHARIC_FONT_STYLE = "font-family: 'Noto Sans Ethiopic', sans-serif;"

# --- Load Haar Cascade ---
haar_cascade_loaded = False
if not os.path.exists(HAAR_CASCADE_PATH):
    FACE_CASCADE = None
    # print("DEBUG: Haar Cascade file not found.")
else:
    FACE_CASCADE = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if FACE_CASCADE.empty():
        FACE_CASCADE = None
        # print("DEBUG: Failed to load Haar Cascade.")
    else:
        haar_cascade_loaded = True
        # print("DEBUG: Haar Cascade loaded successfully.")

# --- API Keys and Model IDs ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
ROBOFLOW_API_KEY = st.secrets.get("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = st.secrets.get("ROBOFLOW_MODEL_ID")
ROBOFLOW_API_URL = st.secrets.get("ROBOFLOW_API_URL", "https://detect.roboflow.com")
# print(f"DEBUG: ROBOFLOW_API_KEY: {'Set' if ROBOFLOW_API_KEY else 'Not Set'}")
# print(f"DEBUG: ROBOFLOW_MODEL_ID: {ROBOFLOW_MODEL_ID}")
# print(f"DEBUG: ROBOFLOW_API_URL: {ROBOFLOW_API_URL}")


# --- Session State ---
if 'user_name' not in st.session_state: st.session_state.user_name = ""
if 'name_submitted' not in st.session_state: st.session_state.name_submitted = False
if 'camera_active' not in st.session_state: st.session_state.camera_active = False
if 'current_emotion_for_user' not in st.session_state: st.session_state.current_emotion_for_user = None
if 'emotion_message_for_user' not in st.session_state: st.session_state.emotion_message_for_user = ""
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'frame_count' not in st.session_state: st.session_state.frame_count = 0
if 'ai_conversation_starter_sent' not in st.session_state: st.session_state.ai_conversation_starter_sent = False


# --- Main App UI ---
st.markdown(f"<h1 style='text-align: center; {AMHARIC_FONT_STYLE}'>የስሜት መመርመሪያ እና AI Chat</h1>", unsafe_allow_html=True)

if not haar_cascade_loaded:
    st.error("የ Haar Cascade መለያን መጫን አልተቻለም። እባክዎ ፋይሉ መኖሩን ያረጋግጡ።")
    st.stop()

if not st.session_state.name_submitted:
    with st.form(key="name_form"):
        name_input = st.text_input("እባክዎ ስምዎን ያስገቡ / Please enter your name:", key="user_name_input_form")
        submit_button = st.form_submit_button(label="አስገባ / Submit")
        if submit_button and name_input:
            st.session_state.user_name = name_input.strip()
            st.session_state.name_submitted = True
            st.rerun()
        elif submit_button and not name_input:
            st.warning("እባክዎ ስምዎን ያስገቡ። / Please enter your name.")
    st.stop()

gemini_model = None
roboflow_client = None

with st.sidebar:
    if st.session_state.user_name:
        st.markdown(f"<h2 style='{AMHARIC_FONT_STYLE}'>እንኳን ደህና መጣህ, {st.session_state.user_name}!</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='{AMHARIC_FONT_STYLE}'>መቆጣጠሪያዎች</h3>", unsafe_allow_html=True)
    if st.button("📷 ካሜራ አብራ/አጥፋ", key="toggle_camera_sidebar_btn"):
        st.session_state.camera_active = not st.session_state.camera_active
        if not st.session_state.camera_active: # If camera is turned off
            st.session_state.ai_conversation_starter_sent = False # Reset starter flag

    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            st.success("Gemini API ተጀምሯል።", icon="✨")
        except Exception as e:
            st.error(f"Gemini API መጀመር አልተቻለም: {e}")
            st.caption("የኤፒአይ ቁልፍዎን እና ግንኙነትዎን ያረጋግጡ።")
            # print(f"DEBUG: Gemini Init Error: {e}")
    else:
        st.warning("የ Gemini API ቁልፍ አልተገኘም።")

    if ROBOFLOW_API_KEY and ROBOFLOW_API_URL and ROBOFLOW_MODEL_ID:
        try:
            roboflow_client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)
            st.success("Roboflow API ተጀምሯል።", icon="🤖")
            # print("DEBUG: Roboflow client initialized.")
        except Exception as e:
            st.error(f"Roboflow API መጀመር አልተቻለም: {e}")
            # print(f"DEBUG: Roboflow Init Error: {e}")
    else:
        if not ROBOFLOW_API_KEY: st.warning("የ Roboflow API ቁልፍ አልተገኘም።")
        if not ROBOFLOW_MODEL_ID: st.warning("የ Roboflow MODEL ID አልተገኘም።")
        st.warning("የስሜት መመርመሪያ በ Roboflow ላይገኝ ይችላል።")
        # print("DEBUG: Roboflow client NOT initialized (missing key, URL, or model_id).")

def translate_emotion_am(emotion_en):
    translations = {
        "angry": "ንዴት", "disgust": "አስጸያፊ", "fear": "ፍርሃት",
        "happy": "ደስታ", "sad": "ሀዘን", "surprise": "መገረም", "surprised": "መገረም",
        "neutral": "ገለልተኛ", None: "አልታወቀም", "": "አልታወቀም", "error": "ስህተት",
        "ስህተት": "ስህተት"
    }
    if emotion_en == "ስህተት": return "ስህተት"
    return translations.get(str(emotion_en).lower() if emotion_en else "", str(emotion_en))

main_col1, main_col2 = st.columns([2, 1.5])
with main_col1:
    if st.session_state.user_name:
        st.markdown(f"<h3 style='text-align: center; {AMHARIC_FONT_STYLE}'>የካሜራ እይታ ለ {st.session_state.user_name}</h3>", unsafe_allow_html=True)
    video_placeholder = st.empty()

with main_col2:
    if st.session_state.user_name:
        st.markdown(f"<h3 style='{AMHARIC_FONT_STYLE}'>የአሁኑ ሁኔታ ለ {st.session_state.user_name}</h3>", unsafe_allow_html=True)
    emotion_text_placeholder = st.empty()
    emotion_message_placeholder = st.empty()
    st.markdown("---")
    st.markdown(f"<h3 style='{AMHARIC_FONT_STYLE}'>AI Chat (በአማርኛ)</h3>", unsafe_allow_html=True)
    
    if gemini_model:
        # Check if AI should initiate conversation based on emotion
        user_current_emotion_str = str(st.session_state.current_emotion_for_user).lower()
        if user_current_emotion_str == 'sad' and not st.session_state.ai_conversation_starter_sent and st.session_state.camera_active:
            try:
                with st.spinner("AI ምላሽ እየሰጠ ነው..."):
                    # AI initiates the conversation because user is sad
                    initial_ai_prompt = (
                        f"ሰላም {st.session_state.user_name}። እንዳዘንክ አስተውያለሁ። "
                        "ስለ ጉዳዩ ማውራት ትፈልጋለህ? ምን እንደሚያስጨንቅህ ንገረኝ፣ ምናልባት ላግዝህ እችላለሁ።"
                        "በአማርኛ ብቻ መልስ።"
                    )
                    # We are not sending this as a user message, but as a direct AI utterance.
                    # For a more natural flow, we could have the AI "ask" this.
                    # Here, we'll just make it the first message from AI.

                    # For Gemini, we typically send a user prompt to get a response.
                    # To make the AI "speak first", we can construct a short history.
                    contextual_prompt_for_ai_starter = [
                        {"role": "user", "content": f"ሰላም AI፣ ስሜ {st.session_state.user_name} ነው። ትንሽ አዝኛለሁ።"}, # Implied user turn
                        {"role": "model", "content": initial_ai_prompt} # This is what we want the model to say as a starter
                    ]
                    # For an actual call, we'd normally send just the user part.
                    # For this "AI starter", we're simulating the AI's response to an implied situation.
                    # Let's simplify: send a prompt to Gemini AS a system instruction to generate this starter.
                    
                    # Simpler approach: AI generates a response to the *situation*
                    prompt_to_generate_starter = (
                        f"አንተ በጣም አዛኝና በአማርኛ የምትናገር AI ነህ። "
                        f"ስሙ {st.session_state.user_name} የሆነ ተጠቃሚ እንዳዘነ አስተውለሃል። "
                        f"ውይይቱን ለመጀመርና ድጋፍ ለመስጠት ምን ትላለህ? ለምሳሌ፣ 'ሰላም {st.session_state.user_name}፣ እንዳዘንክ አስተውያለሁ። ስለ ጉዳዩ ማውራት ትፈልጋለህ?' ልትል ትችላለህ። "
                        "በአማርኛ ብቻ መልስ።"
                    )
                    response = gemini_model.generate_content(prompt_to_generate_starter)
                    ai_starter_message = response.text
                    
                    if ai_starter_message:
                        st.session_state.chat_history.append({"role": "ai", "text": ai_starter_message})
                        st.session_state.ai_conversation_starter_sent = True # Prevent re-sending
                        st.rerun() # Update chat display
            except Exception as e:
                st.error(f"AI ውይይት ማስጀመሪያ ላይ ስህተት: {e}")
                # print(f"DEBUG: AI starter error: {e}")

        chat_container = st.container()
        with chat_container:
            for entry in st.session_state.chat_history:
                role_display = "እርስዎ" if entry["role"] == "user" else "AI"
                st.markdown(f"<p style='{AMHARIC_FONT_STYLE}'><b>{role_display}:</b> {entry['text']}</p>", unsafe_allow_html=True)
        
        user_input_chat = st.text_input("መልእክትዎን በአማርኛ ያስገቡ:", key="chat_input_main_form") # Changed from "ጥያቄዎን" to "መልእክትዎን"
        if st.button("ላክ", key="send_chat_main_button"):
            if user_input_chat:
                st.session_state.chat_history.append({"role": "user", "text": user_input_chat})
                
                # Construct prompt for Gemini, including context if available
                full_prompt_to_gemini = f"ተጠቃሚው {st.session_state.user_name} እንዲህ ይላል: '{user_input_chat}'. "
                if st.session_state.current_emotion_for_user and st.session_state.current_emotion_for_user != "error":
                    user_emotion_am = translate_emotion_am(st.session_state.current_emotion_for_user)
                    if user_emotion_am != "አልታወቀም":
                         full_prompt_to_gemini = (
                            f"አንተ በጣም አዛኝና በአማርኛ የምትናገር AI ነህ። "
                            f"ስሙ {st.session_state.user_name} የሆነ ተጠቃሚ አሁን {user_emotion_am} ስሜት ላይ እንዳለ ታውቃለህ። "
                            f"ተጠቃሚው እንዲህ ይላል: '{user_input_chat}'. "
                            f"ለዚህ መልእክት ስሜቱን ከግምት ውስጥ በማስገባት በአማርኛ ብቻ መልስ።"
                        )
                else: # Generic prompt if no specific emotion
                    full_prompt_to_gemini = (
                        f"አንተ በአማርኛ የምትናገር AI ነህ። "
                        f"ተጠቃሚው {st.session_state.user_name} እንዲህ ይላል: '{user_input_chat}'. "
                        f"በአማርኛ ብቻ መልስ።"
                    )

                try:
                    with st.spinner("AI ምላሽ እየሰጠ ነው..."):
                        response = gemini_model.generate_content(full_prompt_to_gemini)
                        ai_response = response.text
                    st.session_state.chat_history.append({"role": "ai", "text": ai_response})
                    st.session_state.ai_conversation_starter_sent = True # Assume any user interaction means starter is handled
                    st.rerun()
                except Exception as e: 
                    st.error(f"ከ Gemini AI ጋር መገናኘት አልተቻለም: {e}")
                    # print(f"DEBUG: Gemini Chat Error: {e}")
            else: st.warning("እባክዎ መልእክት ያስገቡ።") # Changed from "ጥያቄ"
    else:
        st.warning("የ Gemini API ቁልፍ አልተዋቀረም።")

if st.session_state.camera_active and st.session_state.name_submitted and FACE_CASCADE is not None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("ካሜራውን መክፈት አልተቻለም።")
        st.session_state.camera_active = False
    else:
        # print("DEBUG: Camera opened successfully.")
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                video_placeholder.error("ከካሜራ ፍሬም ማንበብ አልተቻለም።"); break
            frame = cv2.flip(frame, 1)
            rgb_frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_for_haar = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray_for_haar, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            current_emotion_val = st.session_state.current_emotion_for_user
            current_emotion_msg_val = st.session_state.emotion_message_for_user
            run_emotion_detection = (st.session_state.frame_count % 20 == 0)

            if faces is not None and len(faces) > 0:
                x, y, w, h = faces[0]
                cv2.rectangle(rgb_frame_display, (x, y), (x + w, y + h), (255, 0, 0), 2)

                if run_emotion_detection:
                    # print(f"DEBUG: Frame {st.session_state.frame_count}, attempting emotion detection.")
                    if roboflow_client and ROBOFLOW_MODEL_ID:
                        detected_emotion_api = None
                        try:
                            face_roi_bgr = frame[y:y+h, x:x+w]
                            if face_roi_bgr.size > 0:
                                is_success, im_buf_arr = cv2.imencode(".jpg", face_roi_bgr)
                                if is_success:
                                    byte_im = im_buf_arr.tobytes()
                                    # print("DEBUG: Sending image to Roboflow...")
                                    result = roboflow_client.infer(byte_im, model_id=ROBOFLOW_MODEL_ID, confidence=0.4)
                                    # print(f"DEBUG: Roboflow raw result: {result}")

                                    if result and 'predictions' in result and result['predictions']:
                                        sorted_predictions = sorted(result['predictions'], key=lambda p: p.get('confidence', 0), reverse=True)
                                        if sorted_predictions:
                                            detected_emotion_api = sorted_predictions[0].get('class')
                                            # print(f"DEBUG: Detected emotion (obj detect): {detected_emotion_api} conf {sorted_predictions[0].get('confidence')}")
                                    elif result and 'class' in result:
                                        detected_emotion_api = result.get('class')
                                        # print(f"DEBUG: Detected emotion (class): {detected_emotion_api}")
                                    elif result and 'top' in result :
                                        detected_emotion_api = result.get('top')
                                        # print(f"DEBUG: Detected emotion (top): {detected_emotion_api}")
                                    else: # No valid prediction found in result
                                        detected_emotion_api = None 
                                        # print("DEBUG: Roboflow result structure not recognized or no valid predictions.")
                                else: # imencode failed
                                    detected_emotion_api = "error" 
                                    # print("DEBUG: Failed to encode face ROI to JPG.")
                            else: # ROI empty
                                detected_emotion_api = None 
                                # print("DEBUG: Face ROI is empty.")
                            
                            st.session_state.current_emotion_for_user = detected_emotion_api
                            if detected_emotion_api and str(detected_emotion_api).lower() == 'sad':
                                st.session_state.emotion_message_for_user = "ለምን አዘንክ?"
                                # If AI hasn't started convo about sadness and camera is on, it might do so on next UI update
                                # Resetting here allows it to trigger if sadness persists
                                if not st.session_state.ai_conversation_starter_sent:
                                     pass # Let the AI chat section handle it
                            else:
                                st.session_state.emotion_message_for_user = ""
                            
                            current_emotion_val = st.session_state.current_emotion_for_user
                            current_emotion_msg_val = st.session_state.emotion_message_for_user

                        except Exception as e:
                            # print(f"DEBUG: Roboflow emotion detection EXCEPTION: {e}")
                            st.session_state.current_emotion_for_user = "ስህተት"
                            st.session_state.emotion_message_for_user = ""
                            current_emotion_val = "ስህተት"
                            current_emotion_msg_val = ""
                    # else: print("DEBUG: Roboflow client or model ID not available for emotion detection.")
            else: # No faces
                if run_emotion_detection:
                    # print("DEBUG: No faces detected, clearing emotion.")
                    st.session_state.current_emotion_for_user = None
                    st.session_state.emotion_message_for_user = ""
                    current_emotion_val = None
                    current_emotion_msg_val = ""

            video_placeholder.image(rgb_frame_display, channels="RGB", use_container_width=True)
            
            if st.session_state.user_name:
                emotion_display_text = translate_emotion_am(current_emotion_val)
                emotion_text_placeholder.markdown(f"<p style='{AMHARIC_FONT_STYLE}'><b>የ {st.session_state.user_name} ስሜት:</b> {emotion_display_text}</p>", unsafe_allow_html=True)
                if current_emotion_msg_val: # Only show if there's a specific message
                    emotion_message_placeholder.markdown(f"<p style='color:red; {AMHARIC_FONT_STYLE}'>{current_emotion_msg_val}</p>", unsafe_allow_html=True)
                else: # Clear if no specific message (e.g. user is not sad)
                    emotion_message_placeholder.empty()


            st.session_state.frame_count +=1
            time.sleep(0.05) 

        cap.release()
        # print("DEBUG: Camera released.")
        if not st.session_state.camera_active:
             video_placeholder.empty()
             emotion_text_placeholder.empty()
             emotion_message_placeholder.empty()
else:
    if st.session_state.name_submitted and FACE_CASCADE is not None:
        video_placeholder.info(f"{st.session_state.user_name}፣ የቀጥታ የካሜራ እይታን ለማሳየት 'ካሜራ አብራ/አጥፋ' የሚለውን ይጫኑ።")
    elif not st.session_state.name_submitted:
        pass 
    elif FACE_CASCADE is None:
        video_placeholder.error("የፊት መለያ ሞጁል መጫን አልተቻለም። ካሜራ አይገኝም።")
        
    if st.session_state.name_submitted :
        emotion_text_placeholder.empty()
        emotion_message_placeholder.empty()