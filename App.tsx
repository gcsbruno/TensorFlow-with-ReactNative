import { StatusBar } from 'expo-status-bar';
import { Image, View, ActivityIndicator, SafeAreaView } from 'react-native';
import * as ImagePicker from 'expo-image-picker'
import { styles } from './styles';
import { useState } from 'react';
import { Button } from './components/Button';
import * as tensorflow from '@tensorflow/tfjs'
import * as mobilenet from '@tensorflow-models/mobilenet'
import * as FileSystem from 'expo-file-system'
import { decodeJpeg } from '@tensorflow/tfjs-react-native'
import { Classification, ClassificationProps } from './components/Classification';

export default function App() {
  const [isLoading, setIsLoading] = useState(false)
  const [selectedImageUri, setSelectedImageUri] = useState('')
  const [results, setResults] = useState<ClassificationProps[]>([]);

  async function handleSelectImage() {
    setIsLoading(true)

    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
      });

      if (!result.canceled) {
        const { uri } = result.assets[0]
        setSelectedImageUri(uri)
        await imageClassification(uri)
      }
    } catch (error) {
      console.log(error)
    } finally {
      setIsLoading(false)
    }
  }

  async function imageClassification(imageUri: string) {
    setResults([]);

    await tensorflow.ready();
    const model = await mobilenet.load();

    const imageBase64 = await FileSystem.readAsStringAsync(imageUri, {
      encoding: FileSystem.EncodingType.Base64
    });

    const imgBuffer = tensorflow.util.encodeString(imageBase64, 'base64').buffer;
    const raw = new Uint8Array(imgBuffer)

    const imageTensor = decodeJpeg(raw);

    const classificationResult = await model.classify(imageTensor);
    setResults(classificationResult)
  }

  return (
    <SafeAreaView style={{ flex: 1 , backgroundColor: '#171717'}}>
      <View style={styles.container}>

        <StatusBar
          style="light"
          backgroundColor='transparent'
          translucent />

        <Image
          source={{ uri: selectedImageUri ? selectedImageUri : 'https://img.freepik.com/premium-vector/photo-icon-picture-icon-image-sign-symbol-vector-illustration_64749-4409.jpg' }}
          style={styles.image}
        />

        <View style={styles.results}>
          {results.map((result) => (
            <Classification key={result.className} data={result} />
          ))
          }
        </View>
        {isLoading
          ? <ActivityIndicator color="#5F1BBF" />
          : <Button title="Selecionar Imagem" onPress={handleSelectImage} />}
      </View>
    </SafeAreaView>
  );
}


