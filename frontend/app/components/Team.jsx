const teamMembers = [
  {
    name: "Giribala",
    role: "Frontend Developer",
    desc: "Crafts intuitive and accessible interfaces that ensure a calm, seamless user experience across all devices.",
  },
  {
    name: "Joel",
    role: "Backend Engineer",
    desc: "Builds and maintains the core services powering Lenni — ensuring secure, fast, and scalable infrastructure.",
  },
  {
    name: "Kesiya",
    role: "Machine Learning Engineer",
    desc: "Designs empathetic language models and fine-tunes Lenni's responses to provide human-like support.",
  },
  {
    name: "Kevin",
    role: "ML Research & Ethics",
    desc: "Focuses on responsible AI, ensuring Lenni responds safely, fairly, and transparently in every interaction.",
  },
];

const Team = () => {
  return (
    <section
      id="team"
      className="min-h-screen flex flex-col items-center justify-center px-6 py-16 text-center space-y-8"
    >
      <h1 className="text-4xl font-bold bg-clip-text text-gradient">
        The Team Behind Lenni
      </h1>

      <h2 className="max-w-4xl text-dark-200 text-2xl">
        A small team of engineers, researchers, and designers dedicated to making mental health support more accessible and empathetic through AI.
      </h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-8 max-w-5xl w-full mt-8">
        {teamMembers.map((member, index) => (
          <div key={index} className="rounded-xl shadow-md p-6 text-left border-2 border-gray-400 hover:shadow-lg transition-shadow duration-300">
            <h3 className="text-xl font-semibold text-gray-800">{member.name}</h3>
            <p className="text-black text-lg font-medium">{member.role}</p>
            <p className="mt-2 text-gray-700 text-base">{member.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
};

export default Team;
