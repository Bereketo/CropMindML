import React from "react";
import betty from "./Developers/betty.png";
import fita from "./Developers/fita.jpg";
import tuji from "./Developers/tuji.jpg";
import brook from "./Developers/brook.jpg";
import beki from "./Developers/beki.png";
import emran from "./Developers/emran.png";
import mahalet from './Developers/mahalet.png'
import logo from '../images/logo_dev.png'

const TeamMembers = () => {
  const members = [
    {
      name: "Fita Wegene",
      image: fita,
      description: "Team Leader, UI designer",
      contact: "AAU"
    },
    {
      name: "Bereket Alebachew",
      image: beki,
      description: "File and Folder Designer, Developer",
      contact: "AAU"
    },
    {
      name: "Tuji Sultan",
      image: tuji,
      description: "Developer",
      contact: "MWU"
    },
    {
      name: "Betelhem Zekarias",
      image: betty,
      description: "Developer",
      contact: "AAU"
    },
    {
      name: "Biruk Tafese",
      image: brook,
      description: "Developer",
      contact: "AAU"
    },
    {
      name: "Emran Mohammed",
      image: emran,
      description: "Developer",
      contact: "MWU"
    },
    {
      name: "Mahlet Samuel",
      image: mahalet,
      description: "Developer",
      contact: "AASTU"
    },
    // {
    //   name:"SebilAi",
    //   image: logo,
    //   description: "Project",
    //   contact:"sebilAi@developers.com"
    // }
  ];

  return (
    <div className="bg-slate-100">
      <h1 className="text-4xl font-bold animate-pulse text-[#396E8D] mb-8">
      Team Members
      </h1>
      <div className=" border-2  rounded-lg p-5 w-10/12 m-auto px-5 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 place-items-center justify-center">
        {members.map((member, index) => (
         <div
         key={index}
         className=" rounded-lg  m-auto transform transition-transform hover:scale-105"
       >
         <div draggable="false" className="h-[calc(100% - 2rem)] overflow-hidden rounded-lg"> {/* Adjust height minus padding */}
           <img
             src={member.image}
             alt={member.name}
             className=" w-[200px] object-cover"
           />
         </div>
            <div className="mt-2 text-[#396E8D]">
              <p className="font-semibold text-lg">{member.name}</p>
              <p className="text-sm ">{member.description}</p>
              <p className="text-sm">{member.contact}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TeamMembers;
