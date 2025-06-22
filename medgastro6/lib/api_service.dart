import 'package:http/http.dart' as http;
import 'dart:io';
import 'dart:convert';
import 'config.dart';

class ApiService {
  static Future<String> askQuestion(File imageFile, String question) async {
    var uri = Uri.parse(Config.apiBaseUrl);
    var request = http.MultipartRequest('POST', uri);
    request.fields['question'] = question;
    request.files.add(await http.MultipartFile.fromPath('image', imageFile.path));

    try {
      var response = await request.send();
      var resp = await response.stream.bytesToString();

      try {
        var data = jsonDecode(resp);
        if (data is Map && data.containsKey("answer")) {
          final answer = data["answer"];
          if (answer == null || answer.toString().trim().isEmpty) {
            return "No answer could be determined for this image/question.";
          }
          return answer.toString();
        } else if (data is Map && data.containsKey("error")) {
          return "Backend error: ${data["error"]}";
        } else {
          return "No valid answer received from the model.";
        }
      } catch (e) {
        return "Invalid response from server.";
      }
    } catch (e) {
      return "Error generating response from the model.";
    }
  }
}