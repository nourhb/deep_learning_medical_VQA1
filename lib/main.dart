import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import 'log_in_page.dart';
import 'sign_up_page.dart';
import 'forgot_password_page.dart';
import 'home_page.dart';
import 'auth_wrapper.dart';
import 'theme_provider.dart';
import 'notification_service.dart';
import 'config.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await Firebase.initializeApp(
    options: FirebaseOptions(
      apiKey: Config.firebaseConfig['apiKey']!,
      authDomain: Config.firebaseConfig['authDomain']!,
      projectId: Config.firebaseConfig['projectId']!,
      storageBucket: Config.firebaseConfig['storageBucket']!,
      messagingSenderId: Config.firebaseConfig['messagingSenderId']!,
      appId: Config.firebaseConfig['appId']!,
      measurementId: Config.firebaseConfig['measurementId']!,
    ),
  );
  await NotificationService.initialize();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => ThemeProvider(),
      child: Consumer<ThemeProvider>(
        builder: (context, themeProvider, child) {
          return MaterialApp(
            title: 'MedGastro',
            theme: ThemeData(
              primarySwatch: Colors.blue,
              textTheme: GoogleFonts.poppinsTextTheme(Theme.of(context).textTheme),
              colorScheme: ColorScheme.fromSwatch()
                  .copyWith(primary: Color(0xFF1263AF))
                  .copyWith(secondary: Color(0xFF1263AF)),
              brightness: Brightness.light,
              scaffoldBackgroundColor: Color(0xFFF7FAFC),
              cardColor: Colors.white,
              appBarTheme: AppBarTheme(
                backgroundColor: Color(0xFF1263AF),
                foregroundColor: Colors.white,
                elevation: 2,
                titleTextStyle: GoogleFonts.poppins(
                  color: Colors.white,
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                ),
              ),
              elevatedButtonTheme: ElevatedButtonThemeData(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Color(0xFF1263AF),
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  textStyle: GoogleFonts.poppins(fontWeight: FontWeight.w600),
                  elevation: 4,
                ),
              ),
              inputDecorationTheme: InputDecorationTheme(
                filled: true,
                fillColor: Colors.white,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide(color: Color(0xFFB3E5FC)),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide(color: Color(0xFF1263AF), width: 2),
                ),
                labelStyle: GoogleFonts.poppins(color: Color(0xFF1263AF)),
              ),
            ),
            darkTheme: ThemeData(
              brightness: Brightness.dark,
              primarySwatch: Colors.blue,
              textTheme: GoogleFonts.poppinsTextTheme(Theme.of(context).textTheme),
              scaffoldBackgroundColor: Color(0xFF181C20),
              cardColor: Color(0xFF23272B),
              appBarTheme: AppBarTheme(
                backgroundColor: Color(0xFF1263AF),
                foregroundColor: Colors.white,
                elevation: 2,
                titleTextStyle: GoogleFonts.poppins(
                  color: Colors.white,
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                ),
              ),
              elevatedButtonTheme: ElevatedButtonThemeData(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Color(0xFF1263AF),
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  textStyle: GoogleFonts.poppins(fontWeight: FontWeight.w600),
                  elevation: 4,
                ),
              ),
              inputDecorationTheme: InputDecorationTheme(
                filled: true,
                fillColor: Color(0xFF23272B),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide(color: Color(0xFFB3E5FC)),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide(color: Color(0xFF1263AF), width: 2),
                ),
                labelStyle: GoogleFonts.poppins(color: Color(0xFFB3E5FC)),
              ),
            ),
            themeMode: themeProvider.isDarkMode ? ThemeMode.dark : ThemeMode.light,
            debugShowCheckedModeBanner: false,
            home: AuthWrapper(),
            routes: {
              '/login': (context) => LoginPage(),
              '/signup': (context) => SignUpPage(),
              '/forgot-password': (context) => ForgotPasswordPage(),
            },
          );
        },
      ),
    );
  }
}