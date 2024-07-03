import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useState, useRef, useEffect } from 'react';
import io from "socket.io-client";
import { Audio } from 'expo-av';

//when i get to it, function is just     playSound();

const socketEndpoint = "http://100.73.7.50:8000";

export default function App() {
  const [facing, setFacing] = useState('back');
  const [permission, requestPermission] = useCameraPermissions();
  const socketRef = useRef(null);
  const cameraRef = useRef(null);
  const [hasConnection, setConnection] = useState(false);
  const [sound, setSound] = useState();

  useEffect(() => {
    configureAudio();
    const intervalId = setInterval(captureFrame, 2000);
    return () => clearInterval(intervalId);
  }, []);

  async function configureAudio() {
    await Audio.setAudioModeAsync({
      playsInSilentModeIOS: true,
      staysActiveInBackground: true,
      shouldDuckAndroid: true,
    });
  }

  useEffect(function didMount() {
    const socket = io(socketEndpoint, {
      transports: ["websocket"],
    });

    socketRef.current = socket;

    socket.io.on("open", () => setConnection(true));
    socket.io.on("close", () => setConnection(false));
    socket.on("message", (data) => {
      console.log(data);
    });

    return function didUnmount() {
      socket.disconnect();
      socket.removeAllListeners();
    };
  }, []);

  const captureFrame = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.13,
        ImageType: "jpg",
      });
      const response = await fetch(photo.uri);
      const blob = await response.blob();
      console.log("Blob length:", blob.size);
      socketRef.current.emit("message", blob);
    }
  };

  if (!permission) {
    return <View />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: 'center' }}>We need your permission to show the camera</Text>
        <Button onPress={requestPermission} title="grant permission" />
      </View>
    );
  }

  function toggleCameraFacing() {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  }

  async function playSound() {
    console.log("Loading Sound");
    const { sound } = await Audio.Sound.createAsync(
      require('./assets/chime.mp3'),
      { shouldPlay: true }
    );
    setSound(sound);
    console.log("Playing Sound");
    await sound.playAsync();
  }

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