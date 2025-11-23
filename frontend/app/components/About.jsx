const About = () => {
  return (
    <section id="about" className="min-h-screen flex flex-col items-center justify-center px-6 py-16 text-center space-y-8">
      <h1 className="text-4xl font-bold text-gray-800">
        About Your Companion, Lenni
      </h1>

      <h2 className="max-w-6xl">
        Lenni is an AI-powered mental health companion built to provide emotional support, clarity, and a safe space — 24/7. Whether you're feeling anxious, overwhelmed, or just need someone to talk to, Lenni is here to help you feel heard and supported.
      </h2>

      <h3 className="max-sm:text-xl text-2xl max-w-6xl text-dark-200">
        Lenni was created to make mental health support easier to access by removing common barriers like cost, stigma, and long wait times. It offers a safe, judgment-free space where anyone can have a private conversation anytime — no appointments needed.
      </h3>
    </section>
  );
};

export default About;