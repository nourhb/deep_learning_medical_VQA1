import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

import 'home_page.dart';
import 'log_in_page.dart';

class AuthWrapper extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      stream: FirebaseAuth.instance.authStateChanges(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return Center(child: CircularProgressIndicator());
        }
        if (snapshot.hasData) {
          return HomePage(); // ✅ Redirige vers HomePage si connecté
        }
        return LoginPage(); // ✅ Sinon vers LoginPage
      },
    );
  }
}