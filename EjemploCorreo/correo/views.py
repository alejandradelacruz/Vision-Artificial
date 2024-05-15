from django.shortcuts import render, redirect
from django.core.mail import EmailMessage
from .forms import FormularioContacto
from django.http import HttpResponse

def index(request):

    return render(request,"correo/home.html")

def correo(request):

    formulario_contacto=FormularioContacto()

    if request.method=="POST":
        formulario_contacto=FormularioContacto(data=request.POST)
        if formulario_contacto.is_valid():
            nombre=request.POST.get("nombre")
            email=request.POST.get("email")
            contenido=request.POST.get("contenido")

            email=EmailMessage("Mensaje de app Django",
            "El usuario con nombre {} con la dirección {} escribe lo siguiente:\n\n {}".format(nombre, email, contenido), 
            '',
            ["coloca_aqui_tu_email"], 
            reply_to=[email])

            try:
                email.send()

                return redirect("/correo/?valido")
            except:
                return redirect("/correo/?novalido")

    return render(request, "correo/correo.html", {'miFormulario':formulario_contacto})
# Create your views here.

