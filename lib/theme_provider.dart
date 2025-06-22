import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ThemeProvider extends ChangeNotifier {
  static const String themeKey = 'isDarkMode';
  bool _isDarkMode = false;

  bool get isDarkMode => _isDarkMode;

  ThemeProvider() {
    _loadTheme();
  }

  void toggleTheme(bool isOn) async {
    _isDarkMode = isOn;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(themeKey, _isDarkMode);
  }

  void _loadTheme() async {
    final prefs = await SharedPreferences.getInstance();
    _isDarkMode = prefs.getBool(themeKey) ?? false;
    notifyListeners();
  }
} 