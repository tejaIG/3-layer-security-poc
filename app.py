import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from database_schema import init_database, UserAuth
import time
from datetime import datetime

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def check_gesture(hand_landmarks):
    """Check if the user is performing the victory sign gesture"""
    if not hand_landmarks:
        return False
        
    # Get finger tip and pip landmarks for index and middle fingers
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    
    # Check if index and middle fingers are extended while others are closed
    return (index_tip.y < middle_tip.y) and (ring_tip.y > middle_tip.y)

def calculate_orientation(face_landmarks, frame):
    """Calculate head orientation angles from facial landmarks."""
    img_h, img_w = frame.shape[:2]
    
    face_3d = []
    face_2d = []
    
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
    
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    focal_length = 1 * img_w
    cam_matrix = np.array([
        [focal_length, 0, img_h / 2],
        [0, focal_length, img_w / 2],
        [0, 0, 1]
    ])
    
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    success, rot_vec, trans_vec = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix
    )
    
    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    
    return x, y, z

def login_interface():
    st.title("Secure Face Authentication System")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
            <h2 style='color: #1f77b4;'>Login</h2>
        """, unsafe_allow_html=True)
        
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.button("Login", key="login_button")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
            <h3 style='color: #1f77b4;'>Authentication Steps:</h3>
            <ol>
                <li>Enter your credentials</li>
                <li>Show victory sign gesture</li>
                <li>Complete face verification</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    return email, password, submit

def main():
    # Initialize database
    init_database()
    auth = UserAuth()
    
    # Session state initialization
    if 'login_phase' not in st.session_state:
        st.session_state.login_phase = 'credentials'
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    # Login interface
    if st.session_state.login_phase == 'credentials':
        email, password, submit = login_interface()
        
        if submit:
            user_id = auth.verify_user(email, password)
            if user_id:
                st.session_state.user_id = user_id
                st.session_state.login_phase = 'gesture'
                st.rerun()
            else:
                st.error("Invalid credentials!")
    
    # Gesture verification phase
    elif st.session_state.login_phase == 'gesture':
        st.title("Gesture Verification")
        st.info("Please show the victory sign ✌️ to continue")
        
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        gesture_detected = False
        
        try:
            while not gesture_detected:
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = hands.process(rgb_frame)
                
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            rgb_frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )
                        if check_gesture(hand_landmarks):
                            gesture_detected = True
                            st.session_state.login_phase = 'face_verify'
                            break
                
                frame_placeholder.image(rgb_frame, channels="RGB")
                
        finally:
            cap.release()
            if gesture_detected:
                st.rerun()
    
    # Face verification phase
    elif st.session_state.login_phase == 'face_verify':
        st.title("Face Verification")
        st.info("Please look at the camera and move your head slightly")
        
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        orientation_text = st.empty()
        
        vectors_data = []
        verification_time = 5  # seconds
        start_time = time.time()
        
        try:
            while time.time() - start_time < verification_time:
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            rgb_frame,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec
                        )
                        
                        x, y, z = calculate_orientation(face_landmarks, frame)
                        vectors_data.append({
                            'timestamp': datetime.now(),
                            'x': x,
                            'y': y,
                            'z': z
                        })
                        
                        orientation_text.text(f"""
                        Head Orientation:
                        X (Pitch): {x:.2f}°
                        Y (Yaw): {y:.2f}°
                        Z (Roll): {z:.2f}°
                        Time remaining: {verification_time - (time.time() - start_time):.1f}s
                        """)
                
                frame_placeholder.image(rgb_frame, channels="RGB")
            
            # Verify face orientation patterns
            if len(vectors_data) > 0:
                auth.log_login_attempt(
                    st.session_state.user_id,
                    success=True,
                    vectors=vectors_data
                )
                st.success("Login successful!")
                st.session_state.login_phase = 'completed'
            else:
                auth.log_login_attempt(
                    st.session_state.user_id,
                    success=False,
                    failure_reason="No face detected",
                )
                st.error("Face verification failed!")
                st.session_state.login_phase = 'credentials'
                
        finally:
            cap.release()
    
    elif st.session_state.login_phase == 'completed':
        st.title("Welcome!")
        st.success("You have successfully logged in!")
        if st.button("Logout"):
            st.session_state.login_phase = 'credentials'
            st.session_state.user_id = None
            st.rerun()

if __name__ == "__main__":
    main()