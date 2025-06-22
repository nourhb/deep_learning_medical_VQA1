import 'package:flutter/material.dart';
import 'firebase_auth_service.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';

class LoginPage extends StatefulWidget {
  @override
  _LoginPageState createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final FirebaseAuthService _auth = FirebaseAuthService();
  bool _isLoading = false;
  bool _isPasswordObscured = true;
  bool _showResendVerification = false;
  String _unverifiedEmail = '';

  // Validation email
  String? validateEmail(String? value) {
    if (value == null || value.isEmpty) {
      return 'Please enter your email';
    }
    final emailRegex = RegExp(r'^[\w-.]+@([\w-]+\.)+[\w-]{2,4}$');
    if (!emailRegex.hasMatch(value)) {
      return 'Invalid email address';
    }
    return null;
  }

  // Validation mot de passe
  String? validatePassword(String? value) {
    if (value == null || value.isEmpty) {
      return 'Please enter your password';
    }
    if (value.length < 6) {
      return 'Password must be at least 6 characters';
    }
    return null;
  }

  void _submitForm() async {
    if (_formKey.currentState!.validate()) {
      setState(() {
        _isLoading = true;
        _showResendVerification = false;
      });
      final result = await _auth.signInWithEmailAndPassword(
        _emailController.text,
        _passwordController.text,
      );
      if (result.isSuccess) {
        final user = result.value;
        if (user != null) {
          await user.reload();
          if (user.emailVerified) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text('Login successful!')),
            );
            Navigator.pushReplacementNamed(context, '/home');
          } else {
            setState(() {
              _showResendVerification = true;
              _unverifiedEmail = user.email ?? '';
            });
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text('Please verify your email before logging in.')),
            );
            await _auth.signOut();
          }
        }
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(result.error ?? 'Unknown error')),
        );
      }
      setState(() => _isLoading = false);
    }
  }

  final GoogleSignIn _googleSignIn = GoogleSignIn();
  final FirebaseAuth _firebaseAuth = FirebaseAuth.instance;

  Future<void> _signInWithGoogle() async {
    setState(() => _isLoading = true);

    try {
      final GoogleSignInAccount? googleUser = await _googleSignIn.signIn();
      if (googleUser == null) return; // Annulation par l'utilisateur

      final GoogleSignInAuthentication googleAuth = await googleUser.authentication;
      final OAuthCredential credential = GoogleAuthProvider.credential(
        accessToken: googleAuth.accessToken,
        idToken: googleAuth.idToken,
      );

      final UserCredential result = await _firebaseAuth.signInWithCredential(credential);
      final User? user = result.user;

      if (user != null) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Login successful!')),
        );
        Navigator.pushReplacementNamed(context, '/home');
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Google sign-in failed: $e')),
      );
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _resendVerificationEmail() async {
    setState(() => _isLoading = true);
    try {
      final user = await _firebaseAuth.signInWithEmailAndPassword(
        email: _unverifiedEmail,
        password: _passwordController.text,
      );
      await user.user?.sendEmailVerification();
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Verification email resent. Please check your inbox.')),
      );
      await _firebaseAuth.signOut();
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to resend verification email: $e')),
      );
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      body: SingleChildScrollView(
        padding: EdgeInsets.zero,
        child: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [
                Theme.of(context).scaffoldBackgroundColor,
                Theme.of(context).scaffoldBackgroundColor,
                Theme.of(context).scaffoldBackgroundColor,
              ],
            ),
          ),
          child: SafeArea(
            child: Column(
              children: [
                // App Bar personnalisé avec gradient
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
                  child: Text(
                    'Welcome to MedGastro',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      shadows: [
                        Shadow(
                          color: Colors.black.withOpacity(0.3),
                          blurRadius: 4,
                          offset: Offset(0, 2),
                        )
                      ],
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),

                // Logo avec taille adaptée
                Container(
                  margin: EdgeInsets.symmetric(vertical: 20),
                  padding: EdgeInsets.all(15),
                  decoration: BoxDecoration(
                    color: Theme.of(context).cardColor,
                    borderRadius: BorderRadius.circular(20),
                    boxShadow: [
                      BoxShadow(
                        color: Theme.of(context).shadowColor.withOpacity(0.1),
                        blurRadius: 10,
                        offset: Offset(0, 3),
                      )
                    ],
                  ),
                  child: Column(
                    children: [
                      Image.asset(
                        'assets/images/logo.png',
                        height: 80,
                        fit: BoxFit.contain,
                      ),
                      SizedBox(height: 10),
                      Text(
                        'Your Medical Visual Q&A',
                        style: TextStyle(
                          color: Color(0xFF1263AF),
                          fontWeight: FontWeight.bold,
                          fontSize: 18,
                        ),
                      ),
                    ],
                  ),
                ),

                // Formulaire
                Container(
                  padding: EdgeInsets.symmetric(horizontal: 20),
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
                  child: Form(
                    key: _formKey,
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.start,
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        TextFormField(
                          controller: _emailController,
                          keyboardType: TextInputType.emailAddress,
                          decoration: InputDecoration(
                            labelText: 'Email',
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
                        SizedBox(height: 20),
                        TextFormField(
                          controller: _passwordController,
                          obscureText: _isPasswordObscured,
                          decoration: InputDecoration(
                            labelText: 'Password',
                            labelStyle: TextStyle(color: Color(0xFF1263AF)),
                            prefixIcon: Icon(Icons.lock, color: Color(0xFF1263AF)),
                            suffixIcon: IconButton(
                              icon: Icon(
                                _isPasswordObscured ? Icons.visibility_off : Icons.visibility,
                                color: Color(0xFF1263AF),
                              ),
                              onPressed: () {
                                setState(() {
                                  _isPasswordObscured = !_isPasswordObscured;
                                });
                              },
                            ),
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
                          validator: validatePassword,
                        ),
                        SizedBox(height: 30),
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
                            'Log in',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                        SizedBox(height: 20),
                        if (_showResendVerification)
                          Padding(
                            padding: const EdgeInsets.symmetric(vertical: 8.0),
                            child: ElevatedButton.icon(
                              icon: Icon(Icons.email),
                              label: Text('Resend verification email'),
                              onPressed: _isLoading ? null : _resendVerificationEmail,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.orange,
                                foregroundColor: Colors.white,
                                minimumSize: Size(double.infinity, 48),
                              ),
                            ),
                          ),
                        SizedBox(height: 10),
                        TextButton(
                          onPressed: () {
                            Navigator.pushNamed(context, '/forgot-password');
                          },
                          child: Text(
                            'Forgot password?',
                            style: TextStyle(
                              color: Color(0xFF1263AF),
                              fontSize: 16,
                            ),
                          ),
                        ),
                        SizedBox(height: 10),
                        OutlinedButton(
                          onPressed: () {
                            Navigator.pushNamed(context, '/signup');
                          },
                          style: OutlinedButton.styleFrom(
                            foregroundColor: Color(0xFF1263AF),
                            side: BorderSide(color: Color(0xFF1263AF)),
                            padding: EdgeInsets.symmetric(vertical: 12),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                          ),
                          child: Text(
                            'Create an account',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                        SizedBox(height: 20),
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
                          child: ElevatedButton.icon(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.white,
                              foregroundColor: Color(0xFF1263AF),
                              minimumSize: Size(double.infinity, 48),
                              elevation: 2,
                              side: BorderSide(color: Color(0xFF1263AF), width: 1.2),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                            ),
                            icon: Image.asset(
                              'assets/images/Google-Logo.jpeg',
                              height: 24,
                              width: 24,
                              errorBuilder: (context, error, stackTrace) => Icon(Icons.login, color: Color(0xFF1263AF)),
                            ),
                            label: Text(
                              'Sign in with Google',
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 16,
                                color: Color(0xFF1263AF),
                              ),
                            ),
                            onPressed: _isLoading ? null : _signInWithGoogle,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
