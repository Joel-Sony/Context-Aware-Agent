const Support = () => {
  return (
    <section id="support" className="min-h-screen flex flex-col items-center justify-center px-6 py-16 text-center space-y-8">
      <h1 className="text-4xl font-bold bg-clip-text text-gradient">
        Support & Accessibility
      </h1>

      <h2 className="max-w-6xl text-2xl text-dark-200">
        Lenni is designed to be there for you — anytime, anywhere. Whether you're reaching out in a moment of stress or just checking in with yourself, support is always a click away.
      </h2>

      <h3 className="max-sm:text-xl text-2xl max-w-6xl text-dark-200">
        We're committed to making Lenni accessible and inclusive for everyone. If you experience any issues or have suggestions, our team is here to help. Your feedback helps us improve and grow.
      </h3>

      <a href="/start" className="text-2xl w-fit primary-button mt-6 inline-block">
        Talk to Lenni
      </a>
    </section>
  );
};

export default Support;
