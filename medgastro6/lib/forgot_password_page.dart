import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

class ForgotPasswordPage extends StatefulWidget {
  @override
  _ForgotPasswordPageState createState() => _ForgotPasswordPageState();
}

class _ForgotPasswordPageState extends State<ForgotPasswordPage> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  bool _isLoading = false;

  // Validation de l'email
  String? validateEmail(String? value) {
    if (value == null || value.isEmpty) {
      return 'Please enter your email';
    }
    final emailRegex = RegExp(r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$');
    if (!emailRegex.hasMatch(value)) {
      return 'Invalid email address';
    }
    return null;
  }

  // Simuler l'envoi du lien de réinitialisation
  void _submitForm() async {
    if (_formKey.currentState!.validate()) {
      setState(() {
        _isLoading = true;
      });
      try {
        await FirebaseAuth.instance.sendPasswordResetEmail(email: _emailController.text.trim());
        setState(() {
          _isLoading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('A reset link has been sent to ${_emailController.text}')),
        );
      } catch (e) {
        setState(() {
          _isLoading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to send reset link: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Theme.of(context).scaffoldBackgroundColor,
              Theme.of(context).scaffoldBackgroundColor,
              Theme.of(context).scaffoldBackgroundColor,
            ],
            stops: [0.0, 0.5, 1.0],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              // App Bar personnalisé
              Container(
                padding: EdgeInsets.symmetric(vertical: 20, horizontal: 20),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.centerLeft,
                    end: Alignment.centerRight,
                    colors: [
                      Color(0xFF1263AF).withOpacity(0.95),
                      Color(0xFF1263AF).withOpacity(0.85),
                    ],
                  ),
                  borderRadius: BorderRadius.vertical(
                    bottom: Radius.elliptical(MediaQuery.of(context).size.width, 50),
                  ),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    IconButton(
                      icon: Icon(Icons.arrow_back, color: Colors.white),
                      onPressed: Navigator.of(context).pop,
                    ),
                    Text(
                      'Reset Password',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        shadows: [
                          Shadow(
                            color: Colors.black.withOpacity(0.3),
                            blurRadius: 4,
                            offset: Offset(0, 2),
                          )
                        ],
                      ),
                    ),
                    SizedBox(width: 24), // Espace pour aligner le texte au centre
                  ],
                ),
              ),

              // Logo avec taille augmentée
              Container(
                margin: EdgeInsets.symmetric(vertical: 30),
                padding: EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Theme.of(context).cardColor,
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Theme.of(context).shadowColor.withOpacity(0.1),
                      blurRadius: 15,
                      offset: Offset(0, 5),
                    )
                  ],
                ),
                child: SizedBox(
                  height: 200,
                  child: Image.asset(
                    'assets/images/background.png',
                    fit: BoxFit.contain,
                  ),
                ),
              ),

              // Formulaire
              Expanded(
                child: Container(
                  padding: EdgeInsets.symmetric(horizontal: 20, vertical: 25),
                  decoration: BoxDecoration(
                    color: Theme.of(context).cardColor,
                    borderRadius: BorderRadius.vertical(top: Radius.circular(30)),
                    boxShadow: [
                      BoxShadow(
                        color: Theme.of(context).shadowColor.withOpacity(0.05),
                        blurRadius: 10,
                        offset: Offset(0, -5),
                      )
                    ],
                  ),
                  child: SingleChildScrollView(
                    child: Form(
                      key: _formKey,
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.start,
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          // Titre
                          Column(
                            children: [
                              Icon(Icons.lock_reset, size: 50, color: Color(0xFF1263AF)),
                              SizedBox(height: 10),
                              Text(
                                'Recover your account',
                                style: TextStyle(
                                  fontSize: 22,
                                  fontWeight: FontWeight.bold,
                                  color: Color(0xFF1263AF),
                                ),
                                textAlign: TextAlign.center,
                              ),
                              SizedBox(height: 20),
                              Text(
                                'Enter your email address and we will send you a link to reset your password.',
                                style: TextStyle(
                                  fontSize: 16,
                                  color: Colors.grey[700],
                                ),
                                textAlign: TextAlign.center,
                              ),
                            ],
                          ),
                          SizedBox(height: 30),

                          // Email
                          TextFormField(
                            controller: _emailController,
                            keyboardType: TextInputType.emailAddress,
                            decoration: InputDecoration(
                              labelText: 'Email address',
                              labelStyle: TextStyle(color: Color(0xFF1263AF)),
                              prefixIcon: Icon(Icons.email, color: Color(0xFF1263AF)),
                              enabledBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Color(0xFFB3E5FC)),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              focusedBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Color(0xFF1263AF), width: 2),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              filled: true,
                              fillColor: Theme.of(context).inputDecorationTheme.fillColor ?? Theme.of(context).cardColor,
                            ),
                            validator: validateEmail,
                          ),
                          SizedBox(height: 30),

                          // Bouton d'envoi
                          ElevatedButton(
                            onPressed: _isLoading ? null : _submitForm,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Color(0xFF1263AF),
                              foregroundColor: Colors.white,
                              padding: EdgeInsets.symmetric(vertical: 15),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                              elevation: 5,
                              shadowColor: Color(0xFF1263AF).withOpacity(0.3),
                            ),
                            child: _isLoading
                                ? CircularProgressIndicator(color: Colors.white)
                                : Text(
                              'Send reset link',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                          SizedBox(height: 20),

                          // Lien retour
                          TextButton(
                            onPressed: Navigator.of(context).pop,
                            style: TextButton.styleFrom(
                              foregroundColor: Color(0xFF1263AF),
                              padding: EdgeInsets.symmetric(vertical: 10),
                            ),
                            child: Text(
                              'Back to login',
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}