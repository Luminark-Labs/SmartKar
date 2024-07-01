import { CameraView, useCameraPermissions } from 'expo-camera';
import { useState, useRef, useEffect } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import io from "socket.io-client";

// Socket endpoint - replace with your server's IP and port
const socketEndpoint = "http://100.73.7.50:8000";

export default function App() {
  
  // State for camera facing direction
  const [facing, setFacing] = useState('back');
  // Camera permission state
  const [permission, requestPermission] = useCameraPermissions();
  // Refs for socket and camera
  const socketRef = useRef(null);
  const cameraRef = useRef(null);
  // State to track socket connection status
  const [hasConnection, setConnection] = useState(false);

  // Effect to set up a loop for capturing frames every 2 seconds
  useEffect(() => {
    const intervalId = setInterval(captureFrame, 2000);
    return () => clearInterval(intervalId);
  }, []);
  
  // Effect to initialize socket connection and set up event listeners
  useEffect(function didMount() {
    const socket = io(socketEndpoint, {
      transports: ["websocket"],
    });

    socketRef.current = socket;

    // Event listeners for socket connection status
    socket.io.on("open", () => setConnection(true));
    socket.io.on("close", () => setConnection(false));
    // Initial connection message listener
    socket.on("message", (data) => {
      console.log(data);
    });

    // Cleanup function to disconnect socket and remove listeners
    return function didUnmount() {
      socket.disconnect();
      socket.removeAllListeners();
    };
  }, []);

  // Function to capture a frame from the camera and send it via socket
  const captureFrame = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.1,
         
        });
      const response = await fetch(photo.uri);
      const blob = await response.blob();
      console.log("Blob length:", blob.size);
      console.log("image rec");
      socketRef.current.emit("message", blob);
    }
  };

  // Render nothing if permission state is not determined
  if (!permission) {
    return <View />;
  }

  // Render permission request view if camera access is not granted
  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: 'center' }}>We need your permission to show the camera</Text>
        <Button onPress={requestPermission} title="grant permission" />
      </View>
    );
  }

  // Function to toggle camera facing direction
  function toggleCameraFacing() {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  }

  // Main component render function
  return (
    <View style={styles.container}>
      <CameraView 
        style={styles.camera} 
        facing={facing}
        ref={cameraRef}
      >
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={toggleCameraFacing}>
            <Text style={styles.text}>Flip Camera</Text>
          </TouchableOpacity>
        </View>
      </CameraView>
    </View>
  );
}

// StyleSheet for the component
const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: 'transparent',
    margin: 64,
  },
  button: {
    flex: 1,
    alignSelf: 'flex-end',
    alignItems: 'center',
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
});