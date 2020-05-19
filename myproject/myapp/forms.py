from django import forms

class ContactForm(forms.Form):
    brand=forms.CharField()
    product_type=forms.CharField()
    retail_price=forms.CharField()
    discounted_price=forms.CharField()
    



