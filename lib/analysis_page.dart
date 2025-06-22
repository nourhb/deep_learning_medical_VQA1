import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:flutter/foundation.dart'; // Pour Platform.isWeb
import 'api_service.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'firestore_service.dart';

class AnalysisPage extends StatefulWidget {
  @override
  _AnalysisPageState createState() => _AnalysisPageState();
}

class _AnalysisPageState extends State<AnalysisPage> {
  final _formKey = GlobalKey<FormState>();
  final _questionController = TextEditingController();
  final _firestoreService = FirestoreService();
  dynamic _selectedImage; // Peut être File (mobile) ou Uint8List (web)
  String _answer = '';
  bool _isLoading = false;

  final ImagePicker _picker = ImagePicker();

  // Load test image from assets
  Future<void> _loadTestImage() async {
    try {
      final ByteData data = await rootBundle.load('assets/images/test.jpeg');
      final List<int> bytes = data.buffer.asUint8List();
      final tempDir = Directory.systemTemp;
      final file = File('${tempDir.path}/test.jpeg');
      await file.writeAsBytes(bytes);
      
      setState(() {
        _selectedImage = file;
      });
      
      await _firestoreService.addHistoryEntry(
        operationType: 'test_image_load',
        description: 'Test image loaded from assets',
      );
      
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Test image loaded successfully')),
      );
    } catch (e) {
      print('Error loading test image: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error loading test image')),
      );
    }
  }

  // Méthode pour choisir l'image (depuis galerie ou caméra)
  Future<void> _pickImageFrom(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        imageQuality: 70,
      );

      if (pickedFile != null) {
        dynamic imageData;

        if (kIsWeb) {
          imageData = await pickedFile.readAsBytes(); // Pour le web
        } else {
          imageData = File(pickedFile.path); // Pour Android/iOS
        }

        setState(() {
          _selectedImage = imageData;
        });

        await _firestoreService.addHistoryEntry(
          operationType: 'image_selection',
          description: 'Image selected from ${source == ImageSource.gallery ? 'gallery' : 'camera'}',
        );
      }
    } catch (e) {
      debugPrint('Error when selecting the image: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: Unable to load the image')),
      );
    }
  }

  void _generateAnswer() async {
    if (_formKey.currentState!.validate() && _selectedImage != null) {
      setState(() => _isLoading = true);
      print('[VQA] Starting answer generation...');
      try {
        String answer;
        if (!kIsWeb) {
          // Use ApiService for mobile/desktop
          File imageFile = _selectedImage;
          print('[VQA] Image file path: \\${imageFile.path}');
          print('[VQA] Question: \\${_questionController.text}');
          answer = await ApiService.askQuestion(imageFile, _questionController.text);
          print('[VQA] API answer: \\${answer}');
        } else {
          // For web, call Flask backend directly
          final String apiUrl = "http://127.0.0.1:5000/predict";
          var request = http.MultipartRequest('POST', Uri.parse(apiUrl));
          print('[VQA] Adding image as bytes');
          request.files.add(
            http.MultipartFile.fromBytes(
              'image',
              _selectedImage,
              filename: 'image.jpg',
            ),
          );
          print('[VQA] Adding question: \\${_questionController.text}');
          request.fields['question'] = _questionController.text;
          print('[VQA] Sending request to: \\${apiUrl}');
          var response = await request.send();
          print('[VQA] Response status code: \\${response.statusCode}');
          final resp = await response.stream.bytesToString();
          print('[VQA] Raw response: \\${resp}');
          final data = jsonDecode(resp);
          print('[VQA] Decoded response: \\${data}');
          answer = data["answer"] ?? "No answer could be determined for this image/question.";
        }

        setState(() {
          _isLoading = false;
          _answer = answer.toString();
          print('[VQA] _answer set to: \\${_answer}');
        });

        // Save to history
        await _firestoreService.saveMedicalRequest(
          _selectedImage is File ? _selectedImage.path : 'web_image',
          _questionController.text,
          _answer,
        );

      } catch (e, stacktrace) {
        print('[VQA] Error: \\${e}');
        print('[VQA] Stacktrace: \\${stacktrace}');
        setState(() {
          _isLoading = false;
          _answer = "❌ Error: $e";
        });

        // Record error in history
        await _firestoreService.addHistoryEntry(
          operationType: 'error',
          description: 'Error during analysis: $e',
          question: _questionController.text,
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Icon(Icons.medical_services, color: Colors.white, size: 28),
            SizedBox(width: 10),
            Text(
              'Medical Analysis',
              style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
            ),
          ],
        ),
        backgroundColor: Color(0xFF1263AF),
        elevation: 3,
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(24),
        child: Card(
          elevation: 6,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Form(
              key: _formKey,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Section d'importation d'image
                  Container(
                    height: 200,
                    decoration: BoxDecoration(
                      color: Theme.of(context).cardColor,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: _selectedImage != null
                            ? Theme.of(context).colorScheme.primary.withOpacity(0.6)
                            : Theme.of(context).dividerColor.withOpacity(0.3),
                        width: 2,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Theme.of(context).shadowColor.withOpacity(0.1),
                          blurRadius: 5,
                          offset: Offset(0, 3),
                        )
                      ],
                    ),
                    child: Center(
                      child: _selectedImage == null
                          ? Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Row(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    ElevatedButton.icon(
                                      onPressed: () => _pickImageFrom(ImageSource.gallery),
                                      icon: Icon(Icons.image, color: Colors.white),
                                      label: Text(" Add image"),
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: Color(0xFF1263AF),
                                        foregroundColor: Colors.white,
                                        padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                                        shape: RoundedRectangleBorder(
                                          borderRadius: BorderRadius.circular(8),
                                        ),
                                      ),
                                    ),
                                    SizedBox(width: 16),
                                    ElevatedButton.icon(
                                      onPressed: () {
                                        if (kIsWeb) {
                                          ScaffoldMessenger.of(context).showSnackBar(
                                            SnackBar(content: Text('On some browsers, you can select Camera as a source in the file picker.')),
                                          );
                                        }
                                        _pickImageFrom(ImageSource.camera);
                                      },
                                      icon: Icon(Icons.camera_alt, color: Colors.white),
                                      label: Text(" Camera"),
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: Color(0xFF1263AF),
                                        foregroundColor: Colors.white,
                                        padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                                        shape: RoundedRectangleBorder(
                                          borderRadius: BorderRadius.circular(8),
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                                SizedBox(height: 16),
                                Icon(Icons.medical_information, color: Color(0xFF1263AF), size: 40),
                                SizedBox(height: 8),
                                Text(
                                  'Import a medical image to begin',
                                  style: TextStyle(color: Color(0xFF1263AF), fontWeight: FontWeight.w600),
                                ),
                              ],
                            )
                          : _buildImagePreview(_selectedImage),
                    ),
                  ),

                  SizedBox(height: 30),

                  // Champ de question
                  TextFormField(
                    controller: _questionController,
                    maxLines: 3,
                    decoration: InputDecoration(
                      labelText: 'Ask your question',
                      hintText: 'Describe your request regarding the image...',
                      labelStyle: TextStyle(color: Color(0xFF1263AF)),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderSide:
                        BorderSide(color: Color(0xFF1263AF), width: 2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    validator: (value) {
                      if (value == null || value.isEmpty) {
                        return 'Please enter your question';
                      }
                      return null;
                    },
                  ),

                  SizedBox(height: 20),

                  // Bouton d'analyse
                  ElevatedButton.icon(
                    onPressed: _generateAnswer,
                    icon: Icon(Icons.check_circle, size: 20),
                    label: Text(' Analyze'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Color(0xFF1263AF),
                      foregroundColor: Colors.white,      // Texte blanc
                      padding: EdgeInsets.symmetric(vertical: 15),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),

                  SizedBox(height: 30),

                  // Zone de réponse
                  Container(
                    padding: EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Theme.of(context).cardColor,
                      borderRadius: BorderRadius.circular(12),
                      boxShadow: [
                        BoxShadow(
                          color: Theme.of(context).shadowColor.withOpacity(0.15),
                          spreadRadius: 1,
                          blurRadius: 8,
                        ),
                      ],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Response:',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 18,
                            color: Color(0xFF1263AF),
                          ),
                        ),
                        SizedBox(height: 10),
                        Text(
                          _answer != null && _answer.toString().isNotEmpty
                              ? _answer.toString()
                              : 'No results to display. Please import an image and ask a question.',
                          style: TextStyle(fontSize: 16),
                        ),
                      ],
                    ),
                  ),

                  // Indicateur de chargement
                  if (_isLoading)
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 20),
                      child: Center(
                        child: CircularProgressIndicator(
                          color: Color(0xFF1263AF),
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  // Helper function to build image preview based on platform
  Widget _buildImagePreview(dynamic image) {
    if (kIsWeb) {
      return Image.memory(
        image as Uint8List,
        fit: BoxFit.cover,
        height: 200,
      );
    } else {
      return Image.file(
        image as File,
        fit: BoxFit.cover,
        height: 200,
      );
    }
  }
}